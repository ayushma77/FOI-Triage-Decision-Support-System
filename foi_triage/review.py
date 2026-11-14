"""Stage 3: the human gate.

The AI produces a ReviewPacket. Only an officer, through ReviewSession, can
record decisions, and finalise() refuses to produce a release while anything
is outstanding. There is no code path that sets a decision automatically.

Rules enforced in code (see tests/test_review_gate.py):
  1. Every routed flag needs an officer decision: redact, release or consult.
  2. Releasing a flagged passage requires a written reason.
  3. Every record in scope needs an officer outcome, flagged or not, because
     the detector misses things. Unflagged does not mean safe.
  4. finalise() raises ReleaseBlocked listing what is outstanding.
  5. Every action is written to an append-only audit log.
"""

import json
from datetime import datetime, timezone

FLAG_ACTIONS = {"redact", "release", "consult"}
RECORD_OUTCOMES = {"release_in_full", "release_with_redactions", "withhold", "out_of_scope"}


class ReleaseBlocked(Exception):
    def __init__(self, outstanding):
        self.outstanding = outstanding
        super().__init__(f"Release blocked: {len(outstanding)} item(s) need an officer decision")


class ReviewSession:
    def __init__(self, packet, officer):
        if not officer:
            raise ValueError("An officer name is required to open a review session")
        self.packet = packet
        self.officer = officer
        self.flag_decisions = {}
        self.record_outcomes = {}
        self.audit = []
        self._log("session_opened", detail=packet["request_id"])

    # ------------------------------------------------------------------
    def _log(self, action, record_id=None, detail=""):
        self.audit.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "officer": self.officer, "action": action,
                           "record_id": record_id, "detail": detail})

    def _routed_flags(self):
        for rec in self.packet["records"]:
            for f in rec["flags"]:
                if f["routed"]:
                    yield rec["record_id"], f

    # ------------------------------------------------------------------
    def decide_flag(self, flag_id, action, reason=""):
        if action not in FLAG_ACTIONS:
            raise ValueError(f"action must be one of {sorted(FLAG_ACTIONS)}")
        match = [(rid, f) for rid, f in self._routed_flags() if f["flag_id"] == flag_id]
        if not match:
            raise KeyError(f"No routed flag {flag_id} in this packet")
        if action == "release" and not reason.strip():
            raise ValueError("Releasing a flagged passage requires a written reason")
        rid, f = match[0]
        self.flag_decisions[flag_id] = {"action": action, "reason": reason, "officer": self.officer}
        self._log(f"flag_{action}", rid, f"{f['label']}: {f['text'][:80]}" + (f" | reason: {reason}" if reason else ""))

    def set_record_outcome(self, record_id, outcome, reason=""):
        if outcome not in RECORD_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(RECORD_OUTCOMES)}")
        rec = next((r for r in self.packet["records"] if r["record_id"] == record_id), None)
        if rec is None:
            raise KeyError(record_id)
        open_flags = [f["flag_id"] for f in rec["flags"] if f["routed"] and f["flag_id"] not in self.flag_decisions]
        if open_flags and outcome != "out_of_scope":
            raise ReleaseBlocked([f"flag {x}" for x in open_flags])
        if outcome == "release_in_full" and any(
                self.flag_decisions.get(f["flag_id"], {}).get("action") == "redact" for f in rec["flags"]):
            raise ValueError("Record has redactions; use release_with_redactions")
        self.record_outcomes[record_id] = {"outcome": outcome, "reason": reason, "officer": self.officer}
        self._log(f"record_{outcome}", record_id, reason)

    # ------------------------------------------------------------------
    def outstanding(self):
        out = [f"flag {f['flag_id']}" for _, f in self._routed_flags() if f["flag_id"] not in self.flag_decisions]
        out += [f"record {r['record_id']}" for r in self.packet["records"] if r["record_id"] not in self.record_outcomes]
        return out

    def finalise(self):
        missing = self.outstanding()
        if missing:
            self._log("finalise_blocked", detail=f"{len(missing)} outstanding")
            raise ReleaseBlocked(missing)
        self._log("finalised", detail=f"{len(self.record_outcomes)} records")
        return {
            "request_id": self.packet["request_id"],
            "signed_off_by": self.officer,
            "record_outcomes": self.record_outcomes,
            "redactions": [k for k, v in self.flag_decisions.items() if v["action"] == "redact"],
            "consultations": [k for k, v in self.flag_decisions.items() if v["action"] == "consult"],
            "audit": self.audit,
        }

    def export_audit(self, path):
        with open(path, "w") as f:
            for row in self.audit:
                f.write(json.dumps(row) + "\n")

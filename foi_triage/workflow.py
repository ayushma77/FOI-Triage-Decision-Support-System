"""Multi-user workflow: applicant -> AI triage -> information officer -> decision-maker -> release.

Roles
  applicant       lodges requests, tracks status, receives the released package. Never sees flags.
  officer         works the case: scope, flag decisions, record outcomes, then sends for approval.
  decision_maker  reviews the officer's recommendation, approves and releases, or returns it with a note.

Statuses
  in_review -> awaiting_approval -> released
                     |
                     +-> returned (back to the officer with a note) -> awaiting_approval ...

Gates enforced here (the interface mirrors them, but this is the authority):
  * submit: every routed flag decided, every in-scope record has an outcome, case not locked.
  * release of a flagged passage needs a written reason.
  * approve/return: decision-maker only, case must be awaiting approval.
  * the AI never calls act(): there is no AI user and no automatic transition past in_review.
"""

import hashlib
import json
import statistics
import secrets
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

USERS = {
    "alex": {"id": "u-alex", "name": "Alex Morgan", "role": "applicant", "password": "demo"},
    "jordan": {"id": "u-jordan", "name": "Jordan Lee", "role": "officer", "password": "demo"},
    "sam": {"id": "u-sam", "name": "Sam Ngo", "role": "decision_maker", "password": "demo"},
}
ROLE_LABEL = {"applicant": "Applicant", "officer": "Information officer", "decision_maker": "Delegated decision-maker"}
FLAG_ACTIONS = {"redact", "release", "consult"}
OUTCOMES = {"release_in_full", "release_with_redactions", "withhold"}
PUBLIC_STATUS = {"in_review": "Being processed", "returned": "Being processed",
                 "awaiting_approval": "Awaiting decision", "released": "Decision released", "withdrawn": "Withdrawn",
                 "clarification": "Your reply is needed"}


class WorkflowError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


CONSULT_DAYS = 14        # reply requested within (an agency setting, not a statutory period)
FAST_SECONDS = 4         # decisions closer together than this count as fast
CONSULT_RESPONSES = {"agrees": "Agrees to release", "objects": "Objects to release", "no_response": "No response by the due date"}


def _ts(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


class Workflow:
    def __init__(self, path, collections, categories, defaults):
        """collections[col] = {"records": {rid: rec}, "flags": {rid: [[i,c,p,s,e], ...]},
        "packets": {pid: {...}}, "retriever": Retriever or None}"""
        self.path = Path(path)
        self.cols = collections
        self.cats = categories
        self.defaults = defaults
        self.lock = threading.Lock()
        self.sessions = {}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"version": 0, "seq": 0, "cases": {}}
        if not self.state["cases"]:
            self._seed()
            self._save()

    # ------------------------------------------------------------ auth
    def login(self, username, password):
        u = USERS.get((username or "").lower().strip())
        if not u or u["password"] != password:
            raise WorkflowError(401, "Wrong username or password")
        token = secrets.token_hex(16)
        self.sessions[token] = username.lower().strip()
        return token, self.public_user(username.lower().strip())

    def user_for(self, token):
        name = self.sessions.get(token or "")
        if not name:
            raise WorkflowError(401, "Sign in first")
        return {"username": name, **{k: v for k, v in USERS[name].items() if k != "password"}}

    @staticmethod
    def public_user(username):
        u = USERS[username]
        return {"username": username, "id": u["id"], "name": u["name"], "role": u["role"], "role_label": ROLE_LABEL[u["role"]]}

    # ------------------------------------------------------------ helpers
    APPLICANTS = ["Priya Sharma","Tom Wilkins","Mia Nguyen","Ben Okafor","Leah Martin","Grace Liu","Daniel Reyes","Hannah Brooks","Omar Haddad","Jess Kelly","Liam O'Connor","Ruby Tan"]

    def _seed(self):
        n = 0
        for col, c in self.cols.items():
            for pid, p in c["packets"].items():
                mine = pid in ("REQ-0003", "SARA-05")
                nm = self.APPLICANTS[n % len(self.APPLICANTS)]
                n += 1
                applicant = {"id": "u-alex", "name": "Alex Morgan"} if mine else {"id": f"u-pub-{n}", "name": nm}
                self.state["cases"][pid] = self._new_case(pid, col, applicant,
                                                          p["text"], p.get("received") or "2026-09-21", pk=pid,
                                                          n_records=len(p["records"]))

    def _new_case(self, cid, col, applicant, text, received, pk=None, records=None, scope=None, n_records=0, lodged_by=None):
        c = {"id": cid, "col": col, "pk": pk, "text": text, "received": received,
             "due": (date.fromisoformat(received) + timedelta(days=30)).isoformat(),
             "applicant": applicant, "lodged_by": lodged_by, "status": "in_review",
             "t": self.defaults["threshold"], "floor": self.defaults["floor"], "use_floor": True,
             "decisions": {}, "outcomes": {}, "out": [], "promoted": [],
             "officer": None, "approver": None, "note": None, "notice": None, "records": records, "scope": scope,
             "events": []}
        c["events"].append({"time": _now(), "who": applicant["name"], "role": "applicant" if not lodged_by else "officer",
                            "action": "Request lodged", "detail": text[:160], "public": True})
        n = n_records or len(records or [])
        c["events"].append({"time": _now(), "who": "AI triage", "role": "system", "action": "AI triage complete",
                            "detail": f"{n} candidate records retrieved and screened for possible exemptions; sent to an officer for human review",
                            "public": True})
        return c

    def _save(self):
        self.state["version"] += 1
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)

    def case(self, cid):
        c = self.state["cases"].get(cid)
        if not c:
            raise WorkflowError(404, f"No case {cid}")
        return c

    def record_ids(self, c):
        if c.get("records") is not None:
            return [r["id"] for r in c["records"]]
        return [r["id"] for r in self.cols[c["col"]]["packets"][c["pk"]]["records"]]

    def _flags(self, c, rid):
        base = self.cols[c["col"]]["flags"].get(rid, [])
        manual = [[m["k"].rsplit(":", 1)[1], m["c"], 1.0, m["s"], m["e"]] for m in c.get("manual", []) if m["rid"] == rid]
        return base + manual if manual else base

    def routed(self, c, rid, f):
        i, cat, p = f[0], f[1], f[2]
        if f"{rid}:{i}" in c["promoted"]:
            return True
        if p >= c["t"] - 1e-9:
            return True
        return c["use_floor"] and self.cats[cat]["high_harm"] and p >= c["floor"] - 1e-9

    def in_scope(self, c):
        return [r for r in self.record_ids(c) if r not in c["out"]]

    def outstanding(self, c):
        out = []
        for rid in self.in_scope(c):
            for f in self._flags(c, rid):
                k = f"{rid}:{f[0]}"
                if self.routed(c, rid, f) and k not in c["decisions"]:
                    out.append(f"flag {k}")
                elif c["decisions"].get(k, {}).get("action") == "consult":
                    out.append(f"consultation {k}")
            if rid not in c["outcomes"]:
                out.append(f"record {rid}")
        return out

    def _event(self, c, user, action, detail="", public=False):
        c["events"].append({"time": _now(), "who": user["name"], "role": user["role"], "action": action,
                            "detail": detail, "public": public})

    # ------------------------------------------------------------ applicant
    def create_case(self, user, text, col):
        if user["role"] not in ("applicant", "officer"):
            raise WorkflowError(403, "Only applicants or officers can lodge a request")
        text = (text or "").strip()
        if len(text) < 12:
            raise WorkflowError(400, "Describe the documents you want in a little more detail")
        if col not in self.cols:
            raise WorkflowError(400, "Unknown collection")
        retr = self.cols[col].get("retriever")
        records, scope = [], None
        if retr is not None:
            scope, results = retr.search(text)
            records = [{"id": r["record_id"], "ret": {k: r[k] for k in ("relevance", "bm25", "semantic", "matched_terms", "reasons")}}
                       for r in results]
        with self.lock:
            self.state["seq"] += 1
            cid = f"APP-{self.state['seq']:04d}"
            applicant = {"id": user["id"], "name": user["name"]} if user["role"] == "applicant" else {"id": "u-phone", "name": "Lodged on behalf of applicant"}
            c = self._new_case(cid, col, applicant, text, date.today().isoformat(), records=records, scope=scope,
                               lodged_by=user["name"] if user["role"] == "officer" else None)
            self.state["cases"][cid] = c
            self._save()
        return c

    # ------------------------------------------------------------ staff actions
    def act(self, cid, user, typ, p):
        with self.lock:
            c = self.case(cid)
            role = user["role"]
            locked = c["status"] in ("awaiting_approval", "released", "withdrawn")
            officer_types = {"decide", "outcome", "reopen", "scope", "threshold", "promote", "submit", "manual_flag", "consult_response"}
            if typ in officer_types:
                if role != "officer":
                    raise WorkflowError(403, "Only the information officer can change the review")
                if locked:
                    raise WorkflowError(409, "This case has been sent for approval and is read-only")
            if typ == "clarify":
                if role != "officer":
                    raise WorkflowError(403, "Only the information officer can ask for clarification")
                if c["status"] not in ("in_review", "returned"):
                    raise WorkflowError(409, "Clarification can only be asked while the case is in review")
                q = (p.get("question") or "").strip()
                if len(q) < 10:
                    raise WorkflowError(400, "Write the question for the applicant")
                c["clarify"] = {"question": q[:800], "asked_by": user["name"], "asked": _now(), "answer": None,
                                "paused_on": date.today().isoformat(), "prev_status": c["status"]}
                c["status"] = "clarification"
                self._event(c, user, "Asked the applicant to clarify", q[:300], public=True)
            elif typ == "reply":
                if role != "applicant" or c["applicant"]["id"] != user["id"]:
                    raise WorkflowError(403, "Only the applicant can reply")
                if c["status"] != "clarification":
                    raise WorkflowError(409, "No clarification is waiting on this request")
                a = (p.get("answer") or "").strip()
                if not a:
                    raise WorkflowError(400, "Write your reply")
                paused = (date.today() - date.fromisoformat(c["clarify"]["paused_on"])).days
                c["due"] = (date.fromisoformat(c["due"]) + timedelta(days=paused)).isoformat()
                c["clarify"]["answer"], c["clarify"]["answered"] = a[:1500], _now()
                c["status"] = c["clarify"].get("prev_status") or "in_review"
                self._event(c, user, "Applicant replied", a[:300] + (f" (clock paused {paused} day(s))" if paused else ""), public=True)
            elif typ == "withdraw":
                if role != "applicant" or c["applicant"]["id"] != user["id"]:
                    raise WorkflowError(403, "Only the applicant can withdraw their request")
                # (withdrawal is allowed at any stage before release)
                if c["status"] in ("released", "withdrawn"):
                    raise WorkflowError(409, "This request is already closed")
                c["status"] = "withdrawn"
                self._event(c, user, "Request withdrawn by applicant", "", public=True)
            elif typ == "decide":
                key, action, reason = p.get("flag"), p.get("action"), (p.get("reason") or "").strip()
                if action not in FLAG_ACTIONS:
                    raise WorkflowError(400, "Action must be redact, release or consult")
                rid, i = key.rsplit(":", 1)
                f = next((f for f in self._flags(c, rid) if str(f[0]) == i), None)
                if f is None or rid not in self.record_ids(c):
                    raise WorkflowError(404, "No such flag in this case")
                if action == "release" and not reason:
                    raise WorkflowError(400, "Releasing a flagged passage requires a written reason")
                cons = c.setdefault("consults", {}).get(key)
                if action == "consult":
                    party = (p.get("party") or "").strip()[:120] or "Person or business named in the record"
                    c["consults"][key] = {"party": party, "sent": date.today().isoformat(),
                                          "due": (date.today() + timedelta(days=CONSULT_DAYS)).isoformat(),
                                          "response": None, "note": "", "by": user["name"]}
                elif cons and not cons.get("response"):
                    raise WorkflowError(409, "Record the consultation response before deciding")
                c["decisions"][key] = {"action": action, "reason": reason, "by": user["name"], "time": _now()}
                c["outcomes"].pop(rid, None)
                after = f" (after consultation: {CONSULT_RESPONSES[cons['response']].lower()})" if cons and action != "consult" else ""
                self._event(c, user, {"redact": "Redact", "release": "Release despite flag", "consult": "Refer for consultation"}[action],
                            f"{rid}: {self.cats[f[1]]['section']}" + (f" | {reason}" if reason else "")
                            + (f" | consulting {c['consults'][key]['party']}" if action == "consult" else after))
            elif typ == "consult_response":
                key, resp = p.get("flag"), p.get("response")
                cons = c.get("consults", {}).get(key)
                if not cons or c["decisions"].get(key, {}).get("action") != "consult":
                    raise WorkflowError(404, "This passage is not out for consultation")
                if resp not in CONSULT_RESPONSES:
                    raise WorkflowError(400, "Response must be agrees, objects or no_response")
                cons.update(response=resp, note=(p.get("note") or "").strip()[:600], answered=_now())
                self._event(c, user, "Consultation response recorded", f"{key}: {cons['party']}: {CONSULT_RESPONSES[resp]}"
                            + (f" | {cons['note']}" if cons["note"] else ""))
            elif typ == "outcome":
                rid, outcome = p.get("record"), p.get("outcome")
                if outcome not in OUTCOMES or rid not in self.in_scope(c):
                    raise WorkflowError(400, "Invalid record or outcome")
                open_ = [f for f in self._flags(c, rid) if self.routed(c, rid, f) and f"{rid}:{f[0]}" not in c["decisions"]]
                if open_:
                    raise WorkflowError(409, f"{len(open_)} flagged passage(s) on {rid} need a decision first")
                if any(c["decisions"].get(f"{rid}:{f[0]}", {}).get("action") == "consult" for f in self._flags(c, rid)):
                    raise WorkflowError(409, f"{rid} has a passage out for consultation; decide it once the response is in")
                if outcome == "release_in_full" and any(c["decisions"].get(f"{rid}:{f[0]}", {}).get("action") == "redact" for f in self._flags(c, rid)):
                    raise WorkflowError(400, "This record has redactions; release it with redactions")
                c["outcomes"][rid] = {"outcome": outcome, "by": user["name"], "time": _now()}
                self._event(c, user, "Record outcome: " + outcome.replace("_", " "), rid)
            elif typ == "reopen":
                c["outcomes"].pop(p.get("record"), None)
                self._event(c, user, "Record reopened", p.get("record", ""))
            elif typ == "scope":
                rid = p.get("record")
                if rid not in self.record_ids(c):
                    raise WorkflowError(404, "Record not in this case")
                if p.get("in"):
                    if rid in c["out"]:
                        c["out"].remove(rid)
                    self._event(c, user, "Added to scope", rid)
                else:
                    if rid not in c["out"]:
                        c["out"].append(rid)
                    self._event(c, user, "Removed from scope", rid)
                c["outcomes"].pop(rid, None)
            elif typ == "threshold":
                c["t"] = min(0.95, max(0.1, round(float(p.get("t", c["t"])) * 20) / 20))
                c["use_floor"] = bool(p.get("use_floor", c["use_floor"]))
                self._event(c, user, f"Flag settings: threshold {c['t']:.2f}, high-harm floor {'on' if c['use_floor'] else 'off'}")
            elif typ == "manual_flag":
                rid, cat = p.get("record"), p.get("category")
                if rid not in self.record_ids(c):
                    raise WorkflowError(404, "Record not in this case")
                if cat not in self.cats:
                    raise WorkflowError(400, "Choose an exemption ground")
                body = self.cols[c["col"]]["records"][rid]["body"]
                st, en = max(0, int(p.get("start", 0))), min(len(body), int(p.get("end", 0)))
                if en <= st:
                    raise WorkflowError(400, "Select some text first")
                c.setdefault("manual", [])
                k = f"{rid}:m{sum(1 for m in c['manual'] if m['rid'] == rid) + 1}"
                c["manual"].append({"k": k, "rid": rid, "s": st, "e": en, "c": cat, "by": user["name"]})
                c["outcomes"].pop(rid, None)
                self._event(c, user, "Flagged a passage the AI missed", f"{rid}: {self.cats[cat]['section']} | {body[st:en][:80]}")
            elif typ == "promote":
                key = p.get("flag")
                if key not in c["promoted"]:
                    c["promoted"].append(key)
                self._event(c, user, "Raised for review", key)
            elif typ == "submit":
                if c["status"] not in ("in_review", "returned"):
                    raise WorkflowError(409, "This case is waiting on the applicant")
                missing = self.outstanding(c)
                if missing:
                    raise WorkflowError(409, f"Cannot send for approval: {len(missing)} item(s) outstanding")
                if not self.in_scope(c):
                    raise WorkflowError(409, "No records in scope")
                c["status"], c["officer"], c["note"] = "awaiting_approval", user["name"], None
                c["officer_note"] = (p.get("comment") or "").strip()[:600]
                c["review"] = self.review_check(c)
                self._event(c, user, "Sent for approval", "", public=True)
            elif typ in ("approve", "return"):
                if role != "decision_maker":
                    raise WorkflowError(403, "Only the delegated decision-maker can approve or return a case")
                if c["status"] != "awaiting_approval":
                    raise WorkflowError(409, "This case is not awaiting approval")
                if c["officer"] == user["name"]:
                    raise WorkflowError(403, "The officer who prepared a case cannot approve it")
                if typ == "approve":
                    rv = c.get("review") or {"sample": [], "checked": {}}
                    left = [k for k in rv["sample"] if k not in rv["checked"]]
                    if left:
                        raise WorkflowError(409, f"Spot-check {len(left)} more decision(s) before approving")
                    if any(not v["agree"] for v in rv["checked"].values()):
                        raise WorkflowError(409, "You disagreed with a spot-checked decision. Return the case to the officer with a note")
                    c["status"], c["approver"] = "released", user["name"]
                    c["notice"] = self.notice(c)
                    self._event(c, user, "Approved and released", p.get("comment", ""), public=True)
                else:
                    note = (p.get("note") or "").strip()
                    if not note:
                        raise WorkflowError(400, "Say what needs to change when returning a case")
                    c["status"], c["note"] = "returned", note
                    self._event(c, user, "Returned to officer", note)
            elif typ == "spot_check":
                if role != "decision_maker":
                    raise WorkflowError(403, "Only the delegated decision-maker can spot-check")
                if c["status"] != "awaiting_approval":
                    raise WorkflowError(409, "This case is not awaiting approval")
                key, rv = p.get("flag"), c.get("review") or {}
                if key not in rv.get("sample", []):
                    raise WorkflowError(404, "That decision is not in the spot-check sample")
                agree, note = bool(p.get("agree")), (p.get("note") or "").strip()[:400]
                if not agree and not note:
                    raise WorkflowError(400, "Say why you disagree")
                rv["checked"][key] = {"agree": agree, "note": note, "by": user["name"], "time": _now()}
                self._event(c, user, "Spot-check: " + ("agreed with officer" if agree else "disagreed with officer"),
                            key + (f" | {note}" if note else ""))
            else:
                raise WorkflowError(400, f"Unknown action {typ}")
            self._save()
            return c

    # ------------------------------------------------------------ review quality
    def review_check(self, c):
        """Guard against rubber-stamping (automation bias). Looks at how the
        officer's decisions were made, and picks a sample the decision-maker
        must independently confirm before approving: every 'release despite
        flag' first, then high-harm grounds, then the rest. More are sampled
        when the review looks hurried or uniform."""
        decs = sorted((d["time"], k, d) for k, d in c["decisions"].items())
        times = [_ts(t) for t, _, _ in decs]
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        fast = sum(g < FAST_SECONDS for g in gaps)
        warnings = []
        if len(gaps) >= 4 and fast / len(gaps) > 0.5:
            warnings.append(f"{fast} of {len(gaps)} decisions came less than {FAST_SECONDS} seconds after the previous one")
        acts = [d["action"] for _, _, d in decs]
        if len(acts) >= 8 and len(set(acts)) == 1:
            warnings.append(f"Every one of {len(acts)} flags got the same decision ({acts[0]})")
        conf = {f"{rid}:{f[0]}": f for rid in self.in_scope(c) for f in self._flags(c, rid)}
        keys = [k for _, k, _ in decs if k in conf]
        order = lambda k: hashlib.sha256((c["id"] + k).encode()).hexdigest()
        rel = sorted((k for k in keys if c["decisions"][k]["action"] == "release"), key=order)
        hh = sorted((k for k in keys if k not in rel and self.cats[conf[k][1]]["high_harm"]), key=order)
        rest = sorted((k for k in keys if k not in rel and k not in hh), key=order)
        n = 5 if warnings else 3
        sample = (rel + hh + rest)[:max(n, min(len(rel), 8))]
        return {"decisions": len(decs), "median_gap": round(statistics.median(gaps), 1) if gaps else None,
                "fast": fast, "warnings": warnings, "sample": sample, "checked": {}}

    # ------------------------------------------------------------ outputs
    def release_package(self, c):
        recs = self.cols[c["col"]]["records"]
        docs = []
        for rid in self.in_scope(c):
            o = c["outcomes"].get(rid, {}).get("outcome")
            r = recs[rid]
            if o == "withhold":
                secs = sorted({self.cats[f[1]]["section"] for f in self._flags(c, rid) if c["decisions"].get(f"{rid}:{f[0]}")})
                docs.append({"id": rid, "subject": r["subject"], "date": r["date"], "outcome": o, "sections": secs, "body": None})
                continue
            spans = sorted([(f[3], f[4], self.cats[f[1]]["section"].split(" ")[0]) for f in self._flags(c, rid)
                            if c["decisions"].get(f"{rid}:{f[0]}", {}).get("action") == "redact"])
            merged = []                      # overlapping redactions become one bar
            for st, en, sec in spans:
                if merged and st <= merged[-1][1]:
                    m = merged[-1]
                    m[1] = max(m[1], en)
                    if sec not in m[2].split("/"):
                        m[2] += "/" + sec
                else:
                    merged.append([st, en, sec])
            body, pos, parts = r["body"], 0, []
            for st, en, sec in merged:
                parts.append({"t": body[pos:st]})
                parts.append({"redacted": sec, "len": en - st})
                pos = en
            parts.append({"t": body[pos:]})
            docs.append({"id": rid, "subject": r["subject"], "date": r["date"], "outcome": o, "parts": parts})
        return docs

    def notice(self, c):
        ids = self.in_scope(c)
        oc = [c["outcomes"].get(r, {}).get("outcome") for r in ids]
        secs = sorted({self.cats[f[1]]["section"] for r in ids for f in self._flags(c, r)
                       if c["decisions"].get(f"{r}:{f[0]}", {}).get("action") == "redact"})
        lines = [f"NOTICE OF DECISION: ACCESS APPLICATION {c['id']}", "Information Act 2002 (NT)", "",
                 f"Your application received on {c['received']} sought: \"{c['text']}\"", "",
                 f"We identified {len(ids)} documents within the scope of your application.",
                 f"- Released in full: {oc.count('release_in_full')}",
                 f"- Released with information removed: {oc.count('release_with_redactions')}",
                 f"- Withheld in full: {oc.count('withhold')}", "",
                 "Information was removed under:", *([f"- {s}" for s in secs] or ["- No provisions applied"]), "",
                 *([f"We consulted {n} third part{'y' if n == 1 else 'ies'} about information concerning them before deciding.", ""]
                   if (n := len(c.get("consults", {}))) else []),
                 "Where an exemption in Part 4 Division 3 applies, the decision-maker considered whether disclosure "
                 "would be contrary to the public interest (s50).", "",
                 "You may apply for internal review of this decision, and then for review by the Information Commissioner.", "",
                 f"Prepared by {c['officer']}, information officer. Approved by {c['approver']}, delegated decision-maker."]
        return "\n".join(lines)

    def view_for(self, user, c):
        if user["role"] != "applicant":
            return c
        v = {"id": c["id"], "col": c["col"], "text": c["text"], "received": c["received"], "due": c["due"],
             "status": c["status"], "public_status": PUBLIC_STATUS[c["status"]],
             "events": [e for e in c["events"] if e.get("public")]}
        if c.get("clarify"):
            v["clarify"] = {k: c["clarify"].get(k) for k in ("question", "asked", "answer", "answered")}
        if c["status"] == "released":
            v["notice"] = c["notice"]
            v["package"] = self.release_package(c)
        return v

    def list_for(self, user):
        cases = self.state["cases"].values()
        if user["role"] == "applicant":
            cases = [c for c in cases if c["applicant"]["id"] == user["id"]]
        return {"version": self.state["version"], "cases": [self.view_for(user, c) for c in cases]}

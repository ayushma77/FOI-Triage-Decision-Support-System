"""Learning from officers: turn passages they flagged that the AI missed into
candidate detection rules.

Nothing here changes the detector by itself. It writes suggestions that a
person reviews, tests on the held-out set, and only then copies into
sensitivity.RULES. That keeps a human in charge of what the AI looks for, as
well as what gets released.

    python -m foi_triage.feedback                      # reads outputs/workflow.json
    python -m foi_triage.feedback path/to/workflow.json
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOP = set("a an and the of to in on for at by with from about is are was were be this that these those or as it its into "
           "has have had will would should can could our their his her they them you your we he she not but if so".split())
GAP = r"[^.\n]{0,40}?"


def suggest_pattern(text, max_terms=4):
    """Generalise a flagged passage into a regular expression.

    Keeps the first few content words in order and turns digit runs into
    \\d{n}, so "lock combination for the evidence room is 4471" becomes
    lock[^.]{0,40}?combination[^.]{0,40}?evidence[^.]{0,40}?room ... and also
    catches the same marker with a different number or slightly different
    wording in between.
    """
    keep = []
    for tok in re.findall(r"\d+|[A-Za-z][A-Za-z'-]*", text):
        if tok.isdigit():
            keep.append(r"\d{%d}" % len(tok))
        elif len(tok) >= 4 and tok.lower() not in STOP:
            keep.append(re.escape(tok.lower()))
        if len(keep) == max_terms:
            break
    if not keep:
        return re.escape(text.strip().lower())
    return GAP.join(keep)


def suggestions(state, records_by_col):
    """One suggestion per manual flag, with how many other records it would match."""
    out = []
    for cid, c in state.get("cases", {}).items():
        recs = records_by_col.get(c["col"], {})
        for m in c.get("manual", []):
            r = recs.get(m["rid"])
            if not r:
                continue
            body = r.get("body", r.get("b", ""))
            text = body[m["s"]:m["e"]]
            pat = suggest_pattern(text)
            rx = re.compile(pat, re.I)
            hits = sorted(rid for rid, x in recs.items() if rid != m["rid"] and rx.search(x.get("body", x.get("b", ""))))
            out.append({"case": cid, "record": m["rid"], "category": m["c"], "text": text, "pattern": pat,
                        "decision": (c.get("decisions", {}).get(m["k"]) or {}).get("action"),
                        "flagged_by": m.get("by"), "other_matches": len(hits), "examples": hits[:5]})
    return out


def as_rule_lines(items, confidence=0.6):
    """Python lines in the shape sensitivity.RULES uses, ready to review and paste."""
    lines = []
    for s in items:
        why = "Officer-added: " + re.sub(r"\s+", " ", s["text"]).strip()[:50]
        lines.append(f'    ("{s["category"]}", r"{s["pattern"]}", {confidence:.2f}, {json.dumps(why)}),  '
                     f'# {s["case"]} {s["record"]}, matches {s["other_matches"]} other records')
    return lines


def main(path=None):
    path = Path(path or ROOT / "outputs" / "workflow.json")
    if not path.exists():
        print(f"No workflow state at {path}. Run server.py and flag a passage first.")
        return []
    state = json.loads(path.read_text())
    app = json.loads((ROOT / "outputs" / "app_data.json").read_text())
    recs = {col: {rid: {"body": r["b"]} for rid, r in v["records"].items()} for col, v in app["collections"].items()}
    items = suggestions(state, recs)
    if not items:
        print("No officer-added flags yet.")
    else:
        print(f"{len(items)} suggestion(s). Test on the held-out set before adding to sensitivity.RULES:\n")
        print("\n".join(as_rule_lines(items)))
    return items


if __name__ == "__main__":
    main(*sys.argv[1:2])

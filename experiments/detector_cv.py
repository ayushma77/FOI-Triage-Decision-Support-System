"""Choose detector improvements WITHOUT looking at the held-out test phrasings.

Leave-phrasing-out cross-validation on the dev split only: dev phrasings are
divided into 4 folds per category. For each fold, the classifier is trained
on dev records excluding sentences written from that fold's phrasings, then
scored on how many of those unseen-phrasing sentences it routes to the
officer, and how many ordinary sentences it flags by mistake. That mimics
the real question ("does it work on wording it hasn't seen?") inside dev.

The rules were written from all dev phrasings, so CV recall is optimistic in
absolute terms; it is the difference between variants that matters.

    python experiments/detector_cv.py
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foi_triage.corpus import SENSITIVE  # noqa: E402
from foi_triage.pipeline import load_data  # noqa: E402
from foi_triage.sensitivity import SensitivityDetector, is_routed, split_sentences  # noqa: E402

K = 4
MISS_COST = 20   # same costing as the threshold sweep: a miss costs 20 reviews


def template_regex(t):
    parts = re.split(r"\{[a-z_0-9]+\}", t)
    return re.compile(r".+?".join(re.escape(p) for p in parts) + r"$", re.S)


def main():
    records, _, gt, _ = load_data()
    rng = random.Random(7)
    # dev phrasings → fold
    tfold, tre = {}, []
    for cat, items in SENSITIVE.items():
        dev = [t for t, s in items if s == "dev"]
        rng.shuffle(dev)
        for i, t in enumerate(dev):
            tfold[t] = i % K
            tre.append((t, template_regex(t)))
    dev_recs = [r for r in records if gt[r["record_id"]]["split"] == "dev"]
    rfold = {r["record_id"]: i % K for i, r in enumerate(sorted(dev_recs, key=lambda r: r["record_id"]))}

    def tmpl_of(text):
        for t, rx in tre:
            if rx.match(text.strip()):
                return t
        return None

    span_tmpl = {}
    for r in dev_recs:
        for s in gt[r["record_id"]]["spans"]:
            if not s["decoy"]:
                span_tmpl[(r["record_id"], s["start"])] = tmpl_of(s["text"])
    unmatched = sum(v is None for v in span_tmpl.values())

    # Rules are switched off here: they were written from every dev phrasing,
    # so they would "see" the held-out fold. This isolates what is learned.
    variants = {"classifier (current)": (dict(), None, 1)}
    for w in (1, 4, 8):
        variants[f"+ word vectors x{w}"] = (dict(use_vectors=True), None, w)
    for c in (0.3, 0.4, 0.5):
        variants[f"+ names {c}"] = (dict(use_ner=True), c, 1)
        for w in (4, 8):
            variants[f"+ names {c} + vectors x{w}"] = (dict(use_ner=True, use_vectors=True), c, w)
    results = {}
    import foi_triage.sensitivity as S
    for name, (opts, nconf, vw) in variants.items():
        S.VEC_WEIGHT = vw
        if nconf is not None:
            S.NAME_CONF = nconf
        tot = dict(pos=0, hit=0, neg=0, fp=0)
        for f in range(K):
            # training: dev records not in this record fold; drop sentences from this fold's phrasings
            excl = {(rid, st) for (rid, st), t in span_tmpl.items() if t and tfold[t] == f}
            train = [r for r in dev_recs if rfold[r["record_id"]] != f]
            det = SensitivityDetector(**opts).fit(train, gt, split="dev", exclude_spans=excl)
            for r in dev_recs:
                if rfold[r["record_id"]] != f:
                    continue
                gold = [s for s in gt[r["record_id"]]["spans"] if not s["decoy"]]
                flags = [x for x in det.detect(r, use_rules=False) if is_routed(x)]
                for s in gold:
                    t = span_tmpl.get((r["record_id"], s["start"]))
                    if t and tfold[t] == f:
                        tot["pos"] += 1
                        tot["hit"] += any(x["start"] < s["end"] and s["start"] < x["end"] for x in flags)
                for a, b, _ in split_sentences(r["body"]):
                    if not any(a < s["end"] and s["start"] < b for s in gt[r["record_id"]]["spans"]):
                        tot["neg"] += 1
                        tot["fp"] += any(x["start"] < b and a < x["end"] for x in flags)
        results[name] = {**tot, "recall": round(tot["hit"] / tot["pos"], 3), "false_flag_rate": round(tot["fp"] / tot["neg"], 3),
                         "expected_cost": MISS_COST * (tot["pos"] - tot["hit"]) + tot["hit"] + tot["fp"]}
        print(f"{name:18s} unseen-phrasing recall {tot['hit']}/{tot['pos']} = {results[name]['recall']:.1%}   "
              f"false flags {tot['fp']}/{tot['neg']} = {results[name]['false_flag_rate']:.1%}   cost {results[name]['expected_cost']}")
    best = min(results, key=lambda k: results[k]["expected_cost"])
    print("lowest expected cost:", best)
    out = ROOT / "outputs" / "detector_cv.json"
    out.write_text(json.dumps({"folds": K, "best": best, "miss_cost": MISS_COST, "unmatched_spans": unmatched, "variants": results}, indent=2))
    print(f"(unmatched spans: {unmatched}) → {out}")


if __name__ == "__main__":
    main()

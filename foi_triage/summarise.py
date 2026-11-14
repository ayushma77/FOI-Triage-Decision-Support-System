"""Set summary for a request: what was found, what needs a human, and a few
key sentences. Extractive only: every sentence is copied verbatim from a
cited record, so the officer can check it. Sentences carrying a routed flag
are withheld from the summary until the officer has decided on them."""

from collections import Counter

from .exemptions import CATEGORIES
from .retrieval import tokenize
from .sensitivity import is_routed, split_sentences


def summarise(parsed, results, records_by_id, flags_by_id, threshold, floor, max_points=6):
    ids = [r["record_id"] for r in results]
    recs = [records_by_id[i] for i in ids]
    routed = {i: [f for f in flags_by_id[i] if is_routed(f, threshold, floor)] for i in ids}
    cats = Counter(f["category"] for i in ids for f in routed[i])
    dates = sorted(r["date"] for r in recs)
    qterms = set(parsed["topic_terms"])

    points = []
    for r in results[:max_points]:
        rec = records_by_id[r["record_id"]]
        best, best_score = None, -1
        for a, b, sent in split_sentences(rec["body"]):
            if any(f["start"] < b and a < f["end"] for f in routed[rec["record_id"]]):
                continue  # withheld pending review
            sc = len(qterms & set(tokenize(sent)))
            if sc > best_score:
                best, best_score = sent, sc
        if best:
            points.append({"record_id": rec["record_id"], "sentence": best})

    return {
        "records_found": len(ids),
        "records_with_flags": sum(1 for i in ids if routed[i]),
        "flags_to_review": sum(len(v) for v in routed.values()),
        "flags_by_category": {CATEGORIES[c]["label"]: n for c, n in cats.most_common()},
        "date_range": [dates[0], dates[-1]] if dates else None,
        "agencies": dict(Counter(r["agency"] for r in recs)),
        "doc_types": dict(Counter(r["doc_type"] for r in recs)),
        "key_points": points,
        "note": "Extractive summary. Sentences with flagged content are withheld until an officer decides.",
    }

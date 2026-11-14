"""Scores the pipeline against the answer key.

Retrieval: for each request, precision and recall of the returned set,
precision@10, and average precision (rank quality).

Sensitivity is measured on sentences, the unit an officer redacts:
  caught     = a gold sensitive sentence overlaps at least one routed flag
  missed     = a gold sensitive sentence with no routed flag (the privacy-breach error)
  false flag = a routed flag on a sentence that is not sensitive (the workload error)
Reported separately for DEV records (phrasings the rules and classifier were
built from) and TEST records (held-out phrasings). TEST is the honest number.

Error costs: a threshold sweep with and without the high-harm floor. Expected
cost = miss_cost * missed + review_cost * flags. The 20:1 default ratio is an
assumption to be discussed with information officers, not a fact.
"""

from collections import defaultdict

from .exemptions import CATEGORIES
from .sensitivity import DEFAULT_FLOOR, DEFAULT_THRESHOLD, is_routed

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def retrieval_metrics(retriever, requests, gt_requests):
    rows = []
    for req in requests:
        g = gt_requests[req["request_id"]]
        truth = set(g["responsive_record_ids"])
        parsed, res = retriever.search(req["request_text"])
        ids = [x["record_id"] for x in res]
        parse_ok = {
            "dates": parsed["date_from"] == g["date_from"] and parsed["date_to"] == g["date_to"],
            "agency": parsed["agency"] == g["agency"],
            "doc_type": parsed["doc_type"] == g["doc_type"],
        }
        if not truth:
            continue
        tp = len(truth & set(ids))
        hits, ap = 0, 0.0
        for k, i in enumerate(ids, 1):
            if i in truth:
                hits += 1
                ap += hits / k
        rows.append({
            "request_id": req["request_id"], "n_truth": len(truth), "n_returned": len(ids),
            "precision": tp / len(ids) if ids else 0.0, "recall": tp / len(truth),
            "p_at_10": len(truth & set(ids[:10])) / min(10, len(truth)),
            "avg_precision": ap / len(truth), "parse": parse_ok,
        })
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    return {
        "n_requests": n,
        "precision": mean("precision"), "recall": mean("recall"),
        "p_at_10": mean("p_at_10"), "map": mean("avg_precision"),
        "avg_returned": mean("n_returned"),
        "parse_accuracy": {f: sum(r["parse"][f] for r in rows) / n for f in ("dates", "agency", "doc_type")},
        "per_request": rows,
    }


def sensitivity_counts(records, gt_records, flags_by_id, split, threshold, floor, use_floor=True):
    caught = missed = false_flags = flags = 0
    rec_total = rec_caught = 0
    per_cat = defaultdict(lambda: [0, 0])
    cat_correct = 0
    missed_items = []
    for r in records:
        g = gt_records[r["record_id"]]
        if split != "all" and g["split"] != split:
            continue
        routed = [f for f in flags_by_id[r["record_id"]] if is_routed(f, threshold, floor, use_floor)]
        gold = [s for s in g["spans"] if not s["decoy"]]
        flags += len(routed)
        false_flags += sum(1 for f in routed if not any(_overlap(f, s) for s in gold))
        if gold:
            rec_total += 1
            rec_caught += any(any(_overlap(f, s) for f in routed) for s in gold)
        for s in gold:
            hit = [f for f in routed if _overlap(f, s)]
            per_cat[s["category"]][1] += 1
            if hit:
                caught += 1
                per_cat[s["category"]][0] += 1
                cat_correct += max(hit, key=lambda f: f["confidence"])["category"] == s["category"]
            else:
                missed += 1
                missed_items.append({"record_id": r["record_id"], "category": s["category"], "text": s["text"]})
    total = caught + missed
    return {
        "sensitive_sentences": total, "caught": caught, "missed": missed,
        "recall": caught / total if total else 1.0,
        "flags_routed": flags, "false_flags": false_flags,
        "precision": (flags - false_flags) / flags if flags else 1.0,
        "record_recall": rec_caught / rec_total if rec_total else 1.0,
        "category_accuracy_when_caught": cat_correct / caught if caught else 0.0,
        "per_category_recall": {CATEGORIES[c]["label"]: round(a / b, 3) for c, (a, b) in sorted(per_cat.items())},
        "missed_items": missed_items,
    }


def threshold_sweep(records, gt_records, flags_by_id, split="test", miss_cost=20, review_cost=1):
    rows = []
    for use_floor in (True, False):
        for t in THRESHOLDS:
            c = sensitivity_counts(records, gt_records, flags_by_id, split, t, DEFAULT_FLOOR, use_floor)
            rows.append({"floor": use_floor, "threshold": t, "caught": c["caught"], "missed": c["missed"],
                         "recall": round(c["recall"], 3), "flags": c["flags_routed"],
                         "false_flags": c["false_flags"], "precision": round(c["precision"], 3),
                         "expected_cost": miss_cost * c["missed"] + review_cost * c["flags_routed"]})
    return rows


def evaluate_all(records, requests, gt_records, gt_requests, retriever, detector, flags_by_id,
                 miss_cost=20, review_cost=1):
    sens = {}
    for split in ("dev", "test", "all"):
        c = sensitivity_counts(records, gt_records, flags_by_id, split, DEFAULT_THRESHOLD, DEFAULT_FLOOR)
        if split != "test":
            c.pop("missed_items")
        sens[split] = c
    sweep = threshold_sweep(records, gt_records, flags_by_id, "test", miss_cost, review_cost)
    best = min((r for r in sweep if r["floor"]), key=lambda r: (r["expected_cost"], -r["threshold"]))
    return {
        "retrieval": retrieval_metrics(retriever, requests, gt_requests),
        "sensitivity": sens,
        "settings": {"threshold": DEFAULT_THRESHOLD, "high_harm_floor": DEFAULT_FLOOR,
                     "classifier_training_sentences": detector.n_train,
                     "miss_cost": miss_cost, "review_cost": review_cost},
        "sweep_test": sweep,
        "lowest_cost_threshold_test": best,
    }


def real_email_check(detector, enron_by_id, labels, split_ids, thresholds=(0.2, 0.3, 0.5, 0.7)):
    """Record-level check on hand-labelled real Enron emails.
    split_ids = {"dev": [...first 40...], "heldout": [...last 40...]}. The
    correspondence pack was written from dev only; heldout is the honest number."""
    lab = {x["record_id"]: x for x in labels}
    out = []
    for split, ids in split_ids.items():
        for packs in ((), ("correspondence",)):
            flags = {i: detector.detect(enron_by_id[i], packs) for i in ids}
            for t in thresholds:
                tp = fp = fn = tn = n_flags = 0
                cat_hits = defaultdict(lambda: [0, 0])
                for i in ids:
                    routed = [f for f in flags[i] if is_routed(f, t, DEFAULT_FLOOR)]
                    n_flags += len(routed)
                    s, p = lab[i]["sensitive"], bool(routed)
                    tp += s and p
                    fn += s and not p
                    fp += (not s) and p
                    tn += (not s) and not p
                    for c in lab[i]["categories"]:
                        cat_hits[c][1] += 1
                        cat_hits[c][0] += any(f["category"] == c for f in routed)
                out.append({"split": split, "packs": list(packs), "threshold": t,
                            "sensitive_caught": tp, "sensitive_total": tp + fn,
                            "false_alarms": fp, "clean_total": fp + tn,
                            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                            "flags_per_email": round(n_flags / len(ids), 2),
                            "category_hits": {k: f"{a}/{b}" for k, (a, b) in cat_hits.items()}})
    return out

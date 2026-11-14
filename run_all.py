"""Run the whole pipeline end to end.

    python run_all.py                 # uses data/ if present, otherwise generates it
    python run_all.py --regenerate    # rebuild the synthetic corpus first
    python run_all.py --miss-cost 50  # change the error-cost assumption

Outputs (in outputs/):
    review_packets.json      what the officer sees for each request
    metrics.json             retrieval + sensitivity scores
    threshold_sweep.csv      error-cost table (held-out phrasings)
    results.md               human-readable results for the report
    real_request_parsing.json  parser run on 54 real FOI requests (disclosure logs + NT Right to Know)
    real_email_check.json    sensitivity check on 80 hand-labelled real Enron emails
    app.html                 the officer interface served by server.py (Jarvis online)
    officer_review.html      same interface opened as a file (Jarvis offline)
    demo_audit_log.jsonl     a scripted review session showing the gate
"""

import argparse
import csv
import json
from pathlib import Path

from foi_triage import corpus
from foi_triage.evaluate import evaluate_all
from foi_triage import real_data
from foi_triage.evaluate import real_email_check, sensitivity_counts
from foi_triage.pipeline import train_detector
from foi_triage.sensitivity import DEFAULT_FLOOR, DEFAULT_THRESHOLD, SensitivityDetector
from foi_triage.pipeline import DATA, REAL, ROOT, build, build_collection, load_data, load_enron, load_jsonl
from foi_triage.retrieval import parse_request
from foi_triage.review import ReleaseBlocked, ReviewSession
from foi_triage.ui import build_app, build_data
from foi_triage.agreement import make_labeler

OUT = ROOT / "outputs"


def pct(x):
    return f"{100 * x:.0f}%"


def compare_md(m):
    d = m.get("detector_compare")
    if not d:
        return []
    b, a = d["before"], d["after"]
    out = ["## Detector improvement: before and after", "",
           f"Before: rules + classifier. After: {d['mode']} (NT records); {d['email_mode']} (email). "
           "Both changes were chosen by leave-phrasing-out cross-validation on dev phrasings only "
           "(experiments/detector_cv.py), then the held-out set was scored once.", "",
           "| Threshold 0.5 | Before | After |", "|---|---|---|",
           f"| NT held-out phrasings caught | {b['test']['caught']}/{b['test']['sensitive_sentences']} | {a['test']['caught']}/{a['test']['sensitive_sentences']} |",
           f"| NT held-out false alarms | {b['test']['false_flags']} | {a['test']['false_flags']} |",
           f"| NT held-out flags sent to officer | {b['test']['flags_routed']} | {a['test']['flags_routed']} |",
           f"| NT dev false alarms | {b['dev']['false_flags']} | {a['dev']['false_flags']} |"]
    for x, y in zip(b["real_email"], a["real_email"]):
        lab = "core + correspondence pack" if x["packs"] else "core rules"
        out.append(f"| Real emails held-out ({lab}): caught / false alarms | {x['sensitive_caught']}/{x['sensitive_total']} · "
                   f"{x['false_alarms']}/{x['clean_total']} | {y['sensitive_caught']}/{y['sensitive_total']} · {y['false_alarms']}/{y['clean_total']} |")
    if d.get("cv"):
        out += ["", "Cross-validation on dev (rules off, unseen phrasings): " + "; ".join(
            f"{k} {v['hit']}/{v['pos']}" for k, v in d["cv"]["variants"].items()) + f". Chosen: {d['cv']['best']}."]
    out += ["", "Word vectors are off for email: on the 40 dev emails they flagged every message, a domain-shift effect.", ""]
    return out


def write_results_md(m, path):
    r, s = m["retrieval"], m["sensitivity"]
    best = m["lowest_cost_threshold_test"]
    lines = [
        "# Results", "",
        "Synthetic corpus: 500 records, 40 requests, 8 exemption categories. "
        "TEST records use phrasings that the rules and classifier never saw.", "",
        "## Retrieval", "",
        "| Metric | Value |", "|---|---|",
        f"| Recall (responsive records returned) | {pct(r['recall'])} |",
        f"| Precision of returned set | {pct(r['precision'])} |",
        f"| Precision@10 | {pct(r['p_at_10'])} |",
        f"| Mean average precision | {r['map']:.2f} |",
        f"| Average records returned per request | {r['avg_returned']:.1f} |",
        f"| Request parsing: dates / agency / doc type | {pct(r['parse_accuracy']['dates'])} / {pct(r['parse_accuracy']['agency'])} / {pct(r['parse_accuracy']['doc_type'])} |",
        "", "v1 baseline: recall 78%, precision 22%.", "",
        "## Sensitivity flagging (threshold "
        f"{m['settings']['threshold']}, high-harm floor {m['settings']['high_harm_floor']})", "",
        "| | Dev phrasings | Held-out phrasings |", "|---|---|---|",
        f"| Sensitive sentences caught | {s['dev']['caught']}/{s['dev']['sensitive_sentences']} ({pct(s['dev']['recall'])}) | {s['test']['caught']}/{s['test']['sensitive_sentences']} ({pct(s['test']['recall'])}) |",
        f"| Sensitive records with at least one flag | {pct(s['dev']['record_recall'])} | {pct(s['test']['record_recall'])} |",
        f"| Flags sent to officer | {s['dev']['flags_routed']} | {s['test']['flags_routed']} |",
        f"| False alarms | {s['dev']['false_flags']} | {s['test']['false_flags']} |",
        f"| Right category when caught | {pct(s['dev']['category_accuracy_when_caught'])} | {pct(s['test']['category_accuracy_when_caught'])} |",
        "",
        "Held-out recall by category: " + ", ".join(f"{k} {pct(v)}" for k, v in s["test"]["per_category_recall"].items()) + ".",
        "",
        "**Reading this honestly.** The detector is near perfect on wording it was built from and weaker on "
        "new wording. That gap is the case for the design: the system narrows and prioritises, the officer reads "
        "every record, and finalise() is blocked until each one has an officer outcome.", "",
        *compare_md(m),
        "## Error costs (held-out phrasings)", "",
        f"Expected cost = {m['settings']['miss_cost']} x missed + {m['settings']['review_cost']} x flags reviewed. "
        f"Lowest-cost threshold with the high-harm floor on: **{best['threshold']}** "
        f"(missed {best['missed']}, flags {best['flags']}, false alarms {best['false_flags']}).", "",
        "| Threshold | Floor | Caught | Missed | Flags | False alarms | Cost |", "|---|---|---|---|---|---|---|",
    ]
    for row in m["sweep_test"]:
        if row["threshold"] in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            lines.append(f"| {row['threshold']} | {'on' if row['floor'] else 'off'} | {row['caught']} | {row['missed']} | "
                         f"{row['flags']} | {row['false_flags']} | {row['expected_cost']} |")
    if "real_email" in m:
        lines += ["", "## Real emails (80 hand-labelled Enron messages)", "",
                  "Labels: one annotator (Claude), record level: does this email contain material an FOI officer "
                  "would need to review? The correspondence rule pack was written from the first 40 only.", "",
                  "| Split | Rule packs | Threshold | Sensitive caught | False alarms | Flags per email |", "|---|---|---|---|---|---|"]
        for x in m["real_email"]:
            lines.append(f"| {x['split']} | {', '.join(x['packs']) or 'core only'} | {x['threshold']} | "
                         f"{x['sensitive_caught']}/{x['sensitive_total']} | {x['false_alarms']}/{x['clean_total']} | {x['flags_per_email']} |")
    lines += ["", "## What the held-out set missed (default setting)", ""]
    seen = set()
    for it in s["test"]["missed_items"]:
        key = it["text"][:60]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {it['category']}: \"{it['text']}\"")
    Path(path).write_text("\n".join(lines) + "\n")


def demo_review(packet, path):
    """Scripted officer session: shows the gate blocking, then passing."""
    s = ReviewSession(packet, officer="Demo Officer")
    log = []
    try:
        s.finalise()
    except ReleaseBlocked as e:
        log.append(f"finalise() blocked: {len(e.outstanding)} outstanding")
    for rec in packet["records"]:
        for f in rec["flags"]:
            if f["routed"]:
                s.decide_flag(f["flag_id"], "redact")
        outcome = "release_with_redactions" if rec["has_routed_flags"] else "release_in_full"
        s.set_record_outcome(rec["record_id"], outcome, "Officer read the full record")
    result = s.finalise()
    log.append(f"finalise() passed: {len(result['redactions'])} redactions across {len(result['record_outcomes'])} records")
    s.export_audit(path)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--miss-cost", type=float, default=20)
    ap.add_argument("--review-cost", type=float, default=1)
    args = ap.parse_args()

    if args.regenerate or not (DATA / "records.jsonl").exists():
        corpus.generate(out_dir=DATA)
        print("Generated synthetic corpus in data/")
    OUT.mkdir(exist_ok=True)

    records, requests, gt_records, gt_requests = load_data()
    packets, retriever, detector, flags_by_id = build(records, requests, gt_records)
    metrics = evaluate_all(records, requests, gt_records, gt_requests, retriever, detector, flags_by_id,
                           args.miss_cost, args.review_cost)

    (OUT / "review_packets.json").write_text(json.dumps(packets, indent=1))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    with open(OUT / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics["sweep_test"][0]))
        w.writeheader()
        w.writerows(metrics["sweep_test"])

    # ---- real data ----
    if not (REAL / "enron_records.jsonl").exists():
        real_data.prepare()
    enron_records, sara_requests = load_enron()
    # Email collection: names on, word vectors off. Chosen on the 40 dev emails only:
    # vectors trained on NT records flagged every dev email (domain shift).
    email_detector = SensitivityDetector(use_ner=True).fit(records, gt_records, split="dev")
    enron_packets, _, enron_flags = build_collection(enron_records, sara_requests, email_detector, ("correspondence",))
    labels = load_jsonl(REAL / "enron_labels_sample.jsonl")
    ids = [x["record_id"] for x in labels]
    enron_by_id = {r["record_id"]: r for r in enron_records}
    metrics["real_email"] = real_email_check(email_detector, enron_by_id, labels, {"dev": ids[:40], "heldout": ids[40:]})
    (OUT / "real_email_check.json").write_text(json.dumps(metrics["real_email"], indent=1))

    # ---- before / after: v5 detector (rules + classifier) vs current ----
    base = train_detector(records, gt_records, improved=False)
    base_flags = {r["record_id"]: base.detect(r) for r in records}
    pick = lambda c: {k: c[k] for k in ("caught", "sensitive_sentences", "false_flags", "flags_routed", "record_recall", "per_category_recall")}
    base_email = real_email_check(base, enron_by_id, labels, {"heldout": ids[40:]}, thresholds=(0.5,))
    cur_email = [x for x in metrics["real_email"] if x["split"] == "heldout" and x["threshold"] == 0.5]
    cv = json.loads((OUT / "detector_cv.json").read_text()) if (OUT / "detector_cv.json").exists() else None
    metrics["detector_compare"] = {
        "mode": detector.mode, "email_mode": email_detector.mode,
        "before": {"test": pick(sensitivity_counts(records, gt_records, base_flags, "test", DEFAULT_THRESHOLD, DEFAULT_FLOOR)),
                   "dev": pick(sensitivity_counts(records, gt_records, base_flags, "dev", DEFAULT_THRESHOLD, DEFAULT_FLOOR)),
                   "real_email": base_email},
        "after": {"test": pick(metrics["sensitivity"]["test"]), "dev": pick(metrics["sensitivity"]["dev"]), "real_email": cur_email},
        "cv": cv,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_results_md(metrics, OUT / "results.md")

    real = json.loads((DATA / "real_requests_sample.json").read_text())
    nt_real = real_data.load_nt_requests()
    parsed_real = [{"source": x["source"], "request_text": x["request_text"], "parsed": parse_request(x["request_text"])}
                   for x in real]
    register = []
    for x in nt_real["requests"]:
        register.append(dict(x, parsed=parse_request(x["title"])))
        parsed_real.append({"source": x["source"], "request_text": x["title"], "parsed": register[-1]["parsed"]})
    (OUT / "real_request_parsing.json").write_text(json.dumps(parsed_real, indent=1))

    demo_log = demo_review(packets[0], OUT / "demo_audit_log.jsonl")
    data = build_data({"packets": packets, "records": records, "flags": flags_by_id},
                      {"packets": enron_packets, "records": enron_records, "flags": enron_flags},
                      metrics, {"meta": {k: v for k, v in nt_real.items() if k != "requests"}, "requests": register})
    (OUT / "app_data.json").write_text(json.dumps(data, separators=(",", ":")))   # read by server.py
    build_app(data, OUT / "app.html")
    build_app(data, OUT / "officer_review.html")
    build_app(data, OUT / "publish_fragment.html", wrap=False)
    make_labeler(OUT / "label_emails.html")       # blind labelling page for a second annotator

    r, s = metrics["retrieval"], metrics["sensitivity"]
    print(f"Retrieval  recall {pct(r['recall'])}  precision {pct(r['precision'])}  P@10 {pct(r['p_at_10'])}  MAP {r['map']:.2f}")
    print(f"Sensitivity (dev)  caught {s['dev']['caught']}/{s['dev']['sensitive_sentences']}  false alarms {s['dev']['false_flags']}")
    print(f"Sensitivity (held-out)  caught {s['test']['caught']}/{s['test']['sensitive_sentences']}  "
          f"false alarms {s['test']['false_flags']}  records flagged {pct(s['test']['record_recall'])}")
    b = metrics["lowest_cost_threshold_test"]
    print(f"Lowest expected cost (held-out, floor on): threshold {b['threshold']}  missed {b['missed']}  flags {b['flags']}")
    he = [x for x in metrics["real_email"] if x["split"] == "heldout" and x["threshold"] == 0.5]
    for x in he:
        print(f"Real emails (held-out, packs={x['packs']})  caught {x['sensitive_caught']}/{x['sensitive_total']}  "
              f"false alarms {x['false_alarms']}/{x['clean_total']}")
    for line in demo_log:
        print("Gate demo:", line)
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()

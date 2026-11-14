"""Builds the officer review packet for each request in a collection:
retrieval + flags + summary. This is everything the AI pipeline contributes;
decisions happen in review.ReviewSession or the officer interface.

Two collections:
  nt     500 synthetic NT Government records, 40 requests, full answer key
  enron  1,500 real Enron emails, 50 real SARA information needs as requests
"""

import json
from pathlib import Path

from .retrieval import Retriever
from .sensitivity import DEFAULT_FLOOR, DEFAULT_THRESHOLD, SensitivityDetector, is_routed
from .summarise import summarise

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REAL = DATA / "real"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_data(data_dir=DATA):
    records = load_jsonl(data_dir / "records.jsonl")
    requests = load_jsonl(data_dir / "requests.jsonl")
    gt_records = {g["record_id"]: g for g in load_jsonl(data_dir / "ground_truth_records.jsonl")}
    gt_requests = {g["request_id"]: g for g in load_jsonl(data_dir / "ground_truth_requests.jsonl")}
    return records, requests, gt_records, gt_requests


def load_enron():
    return load_jsonl(REAL / "enron_records.jsonl"), load_jsonl(REAL / "sara_requests.jsonl")


def train_detector(records, gt_records, improved=True):
    """Ground truth is used ONLY to pick training sentences (dev split).
    improved=True adds person names and word vectors (spaCy en_core_web_md),
    chosen by leave-phrasing-out CV on dev; falls back silently without spaCy."""
    opts = dict(use_ner=True, use_vectors=True) if improved else {}
    return SensitivityDetector(**opts).fit(records, gt_records, split="dev")


def build_collection(records, requests, detector, packs=(), threshold=DEFAULT_THRESHOLD, floor=DEFAULT_FLOOR):
    retriever = Retriever(records)
    by_id = {r["record_id"]: r for r in records}
    flags_by_id = {r["record_id"]: detector.detect(r, packs) for r in records}
    packets = []
    for req in requests:
        parsed, results = retriever.search(req["request_text"])
        recs = []
        for res in results:
            r = by_id[res["record_id"]]
            fl = [dict(f, routed=is_routed(f, threshold, floor)) for f in flags_by_id[r["record_id"]]]
            recs.append({
                "record_id": r["record_id"], "subject": r["subject"], "agency": r["agency"],
                "doc_type": r["doc_type"], "date": r["date"],
                "retrieval": {k: res[k] for k in ("relevance", "bm25", "semantic", "matched_terms", "reasons")},
                "flags": fl,
                "has_routed_flags": any(f["routed"] for f in fl),
                "officer_outcome": None,   # every record needs an officer outcome; the AI never fills it
            })
        packets.append({
            "request_id": req["request_id"], "request_text": req["request_text"],
            "information_need": req.get("information_need"),
            "received": req.get("received"), "parsed_scope": parsed,
            "settings": {"threshold": threshold, "high_harm_floor": floor, "rule_packs": list(packs)},
            "summary": summarise(parsed, results, by_id, flags_by_id, threshold, floor),
            "records": recs,
        })
    return packets, retriever, flags_by_id


def build(records, requests, gt_records, threshold=DEFAULT_THRESHOLD, floor=DEFAULT_FLOOR):
    """NT collection only (kept for tests and the v2 interface)."""
    detector = train_detector(records, gt_records)
    packets, retriever, flags_by_id = build_collection(records, requests, detector, (), threshold, floor)
    return packets, retriever, detector, flags_by_id

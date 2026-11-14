"""Builds the officer interface: one self-contained HTML file with the
pipeline output for both collections embedded.

  outputs/app.html            served by server.py (Jarvis uses /api/claude, /api/tts, /api/stt)
  outputs/officer_review.html same page opened as a file (Jarvis runs offline)
  publish fragment            same page for claude.ai (Jarvis uses the artifact's Claude access)
"""

import json
from pathlib import Path

from .exemptions import CATEGORIES, HIGH_HARM
from .sensitivity import DEFAULT_FLOOR, DEFAULT_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "ui" / "template.html"
MIN_CONF = 0.1


def _slim_records(records, flags_by_id):
    out = {}
    for r in records:
        fl = []
        for f in flags_by_id[r["record_id"]]:
            if f["confidence"] < MIN_CONF:
                continue
            fl.append({"i": int(f["flag_id"].split(":")[1]), "s": f["start"], "e": f["end"], "c": f["category"],
                       "p": f["confidence"], "r": f["rule_conf"], "m": f["ml_conf"], "w": f["reasons"]})
        out[r["record_id"]] = {"sub": r["subject"], "ag": r["agency"], "cc": r.get("cc", []), "ty": r["doc_type"],
                               "dt": r["date"], "fr": r.get("from", ""), "to": r.get("to", ""), "b": r["body"], "f": fl}
    return out


def _slim_packets(packets):
    return [{"id": p["request_id"], "text": p["request_text"], "need": p.get("information_need"),
             "received": p["received"], "scope": p["parsed_scope"], "points": p["summary"]["key_points"],
             "records": [{"id": r["record_id"], "ret": r["retrieval"]} for r in p["records"]]} for p in packets]


def build_data(nt, enron, metrics, register):
    return {
        "collections": {
            "nt": {"label": "NT agencies", "kind": "Government records",
                   "note": "500 fictional NT Government records with an answer key for every sensitive sentence.",
                   "packets": _slim_packets(nt["packets"]), "records": _slim_records(nt["records"], nt["flags"]),
                   "packs": []},
            "enron": {"label": "Enron archive", "kind": "Corporate email",
                      "note": "1,500 real Enron emails (Enron-Spam release). Requests are the 50 real SARA information needs.",
                      "packets": _slim_packets(enron["packets"]), "records": _slim_records(enron["records"], enron["flags"]),
                      "packs": ["correspondence"]},
        },
        "categories": {k: {"label": v["label"], "section": v["section"], "high_harm": k in HIGH_HARM} for k, v in CATEGORIES.items()},
        "defaults": {"threshold": DEFAULT_THRESHOLD, "floor": DEFAULT_FLOOR},
        "sweep": metrics["sweep_test"],
        "eval": {
            "retrieval": {k: metrics["retrieval"][k] for k in ("recall", "precision", "p_at_10", "map", "parse_accuracy")},
            "dev": {k: metrics["sensitivity"]["dev"][k] for k in ("caught", "sensitive_sentences", "false_flags")},
            "test": {k: metrics["sensitivity"]["test"][k] for k in ("caught", "sensitive_sentences", "false_flags", "record_recall", "per_category_recall")},
            "real_email": metrics["real_email"],
            "miss_cost": metrics["settings"]["miss_cost"], "review_cost": metrics["settings"]["review_cost"],
            "compare": metrics.get("detector_compare"),
        },
        "register": register,
    }


def build_app(data, out_path, wrap=True):
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    if wrap:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + html.replace("<style>", "<style>\n[hidden]{display:none!important}", 1)
                .replace("<!--BODY-->", "</head>\n<body>", 1) + "\n</body>\n</html>\n")
    Path(out_path).write_text(html)
    return out_path


# kept for the v2 single-collection interface used by older scripts
def build_ui(packets, records, metrics, out_path, wrap=True):
    raise NotImplementedError("Use build_data + build_app (v3)")

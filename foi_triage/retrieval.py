"""Stage 1: find records responsive to a request.

1. Parse the request into structure: date range, agency, document type, and
   the remaining subject-matter text.
2. Score every record two ways and fuse the rankings:
     - BM25 keyword match with a small synonym table (catches exact terms)
     - LSA semantic similarity (TF-IDF reduced with SVD, trained only on the
       corpus, no internet needed) so "repair delays" can still find a record
       that only says "maintenance backlog"
   Reciprocal rank fusion combines them without needing to calibrate scores.
3. Apply the parsed scope: date range and document type are hard filters,
   agency is a soft boost (copies to other agencies are often responsive).
4. Return a ranked list with a plain-English reason for each record.

Design choice: the cut-off leans toward recall. A responsive record that is
never shown to the officer is a failure the officer cannot see; an extra
irrelevant record costs a few seconds to rule out.
"""

import math
import re
from collections import Counter
from datetime import date

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MON = r"(january|february|march|april|may|june|july|august|september|october|november|december)"

AGENCY_ALIASES = {
    "NT Department of Health": ["department of health", "nt health"],
    "NT Department of Housing, Local Government and Community Development": ["housing, local government", "department of housing", "territory housing"],
    "NT Police, Fire and Emergency Services": ["police, fire and emergency", "nt police", "pfes"],
    "NT Department of Infrastructure, Planning and Logistics": ["infrastructure, planning", "dipl"],
    "NT Treasury": ["treasury"],
}
DOC_TYPE_CUES = [
    (r"\bbriefing( note)?s?\b|\bministerial brief", "Briefing note"),
    (r"\bmemos?\b|\bmemoranda\b", "Memo"),
    (r"\bminutes\b", "Meeting minutes"),
    (r"\bemails?\b", "Email"),
]
# Phrases that name a document type only as part of a broader list ("emails,
# briefings and reports") should not narrow scope. We only treat a type as a
# filter when it is the sole type mentioned.

STOP = set("""a an and the of to in on for at by with from about all any including relating regarding
during records record documents document information is are was were be this that these those or as
it its into held between copies copy correspondence prepared discussing since onward onwards dated
period concerning""".split())

SYNONYMS = {
    "repair": ["maintenance", "fault", "job"], "maintenance": ["repair"], "backlog": ["outstanding", "overdue"],
    "house": ["housing"], "housing": ["house", "tenancy"], "delay": ["backlog", "overdue"],
    "alcohol": ["amp", "liquor", "takeaway"], "liquor": ["alcohol", "licensee", "licensing"],
    "licence": ["licensing", "licensee"], "compliance": ["inspection", "inspector", "check"],
    "enforcement": ["compliance", "breach"], "tenant": ["tenancy"], "dispute": ["mediation", "tribunal", "ntcat"],
    "staffing": ["roster", "vacancy", "recruitment", "staff"], "workforce": ["staffing", "roster", "recruitment"],
    "police": ["station", "officer"], "fire": ["station", "emergency"], "road": ["infrastructure", "causeway", "bridge"],
    "infrastructure": ["road", "bridge", "barge"], "funding": ["allocated", "program", "grant"],
    "grant": ["grants", "acquittal", "applications"], "community": ["communities"],
    "incident": ["patrol", "incidents"], "safety": ["patrol", "incident"], "contractor": ["contractors", "trades"],
}


def _stem(w):
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + rep
    return w


def tokenize(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


SYN = {_stem(k): [_stem(v) for v in vs] for k, vs in SYNONYMS.items()}


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def _end_of_month(y, m):
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def parse_request(text, today=date(2026, 9, 30)):
    """Turn free text into scope fields. Every field may be None; the caller
    falls back to text-only ranking for anything not found."""
    low = text.lower()
    dates = []
    for d, m, y in re.findall(r"(\d{1,2}) " + MON + r",? (\d{4})", low):
        dates.append(date(int(y), MONTHS[m], int(d)))
    for m, d, y in re.findall(MON + r" (\d{1,2}),? (\d{4})", low):       # "September 1 2025"
        dates.append(date(int(y), MONTHS[m], int(d)))
    month_only = []
    if not dates:
        for m, y in re.findall(MON + r" (\d{4})", low):
            month_only.append((int(y), MONTHS[m]))
    date_from = date_to = None
    if len(dates) >= 2:
        date_from, date_to = min(dates), max(dates)
    elif len(month_only) >= 2:
        (y1, m1), (y2, m2) = min(month_only), max(month_only)
        date_from, date_to = date(y1, m1, 1), _end_of_month(y2, m2)
    elif dates or month_only:
        date_from = dates[0] if dates else date(month_only[0][0], month_only[0][1], 1)
        date_to = today  # "since X", "from X onward", "to the present"
    fy = re.search(r"(\d{4})[-–/](\d{2,4}) financial year", low)
    if fy and not date_from:
        y1 = int(fy.group(1))
        date_from, date_to = date(y1, 7, 1), date(y1 + 1, 6, 30)

    agency = None
    for name, aliases in AGENCY_ALIASES.items():
        if name.lower() in low or any(a in low for a in aliases):
            agency = name
            break

    types = [t for pat, t in DOC_TYPE_CUES if re.search(pat, low)]
    doc_type = types[0] if len(types) == 1 else None

    # Strip the scope words so the topic text is what is left.
    topic = re.sub(r"\d{1,2} " + MON + r",? \d{4}|" + MON + r" \d{4}", " ", low)
    if agency:
        for a in [agency.lower()] + AGENCY_ALIASES[agency]:
            topic = topic.replace(a, " ")
    return {"date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "agency": agency, "doc_type": doc_type,
            "topic_terms": [t for t in tokenize(topic) if not t.isdigit()],
            "topic_text": " ".join(w for w in re.findall(r"[a-z]+", topic) if w not in STOP and len(w) > 2)}


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, records, lsa_dims=80, seed=0):
        self.records = records
        self.by_id = {r["record_id"]: r for r in records}
        texts = [f"{r['subject']} {r['body']}" for r in records]
        self.docs = [tokenize(t) for t in texts]
        self.tf = [Counter(d) for d in self.docs]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(lsa_dims, X.shape[1] - 1), random_state=seed)
        Z = self.svd.fit_transform(X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def _bm25(self, terms, k1=1.2, b=0.75):
        scores = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            L = len(self.docs[i])
            s = 0.0
            for t, w in terms.items():
                f = tf.get(t)
                if f:
                    s += w * self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * L / self.avg_len))
            scores[i] = s
        return scores

    def search(self, request_text, rel_cutoff=0.4, max_results=40, k_rrf=60):
        q = parse_request(request_text)
        terms = {t: 1.0 for t in q["topic_terms"]}
        for t in list(terms):
            for s in SYN.get(t, []):
                terms.setdefault(s, 0.5)
        bm = self._bm25(terms)
        qv = self.svd.transform(self.tfidf.transform([" ".join(q["topic_terms"])]))
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        lsa = self.Z @ qv.ravel()

        rank_bm = {i: r for r, i in enumerate(np.argsort(-bm))}
        rank_lsa = {i: r for r, i in enumerate(np.argsort(-lsa))}
        fused = np.array([1 / (k_rrf + rank_bm[i]) + 1 / (k_rrf + rank_lsa[i]) for i in range(len(self.docs))])
        # Require some real evidence: BM25 > 0 or clear semantic similarity.
        evidence = (bm > 0) | (lsa > 0.35)

        results = []
        for i in np.argsort(-fused):
            if not evidence[i]:
                continue
            r = self.records[i]
            reasons, score = [], float(fused[i])
            if q["date_from"] and not (q["date_from"] <= r["date"] <= q["date_to"]):
                continue
            if q["doc_type"] and r["doc_type"] != q["doc_type"]:
                continue
            if q["date_from"]:
                reasons.append(f"dated {r['date']}, inside {q['date_from']} to {q['date_to']}")
            if q["doc_type"]:
                reasons.append(f"document type is {r['doc_type']}")
            if q["agency"]:
                if r["agency"] == q["agency"]:
                    reasons.append("held by the named agency")
                elif q["agency"] in r.get("cc", []):
                    reasons.append("copied to the named agency")
                else:
                    score *= 0.6
                    reasons.append("other agency (score reduced, not excluded)")
            matched = [t for t in terms if self.tf[i].get(t)]
            results.append({"record_id": r["record_id"], "score": score,
                            "bm25": round(float(bm[i]), 3), "semantic": round(float(lsa[i]), 3),
                            "matched_terms": matched[:8], "reasons": reasons})
        results.sort(key=lambda x: -x["score"])
        if results:
            top = results[0]["score"]
            for x in results:
                x["relevance"] = round(x["score"] / top, 3)
            # Keep records that are semantically close to the best match. The
            # cut-off is relative to the top semantic score so it adapts to how
            # specific the request is; rel_cutoff is tuned for high recall.
            top_sem = max(x["semantic"] for x in results)
            results = [x for x in results if x["semantic"] >= rel_cutoff * top_sem]
        return q, results[:max_results]

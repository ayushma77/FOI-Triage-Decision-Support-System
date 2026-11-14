"""Stage 2: flag passages that may fall under an exemption.

Two detectors, combined:

* RULES: regular expressions for markers an officer would recognise
  (CABINET-IN-CONFIDENCE, a date of birth, "legally privileged"). Each rule
  carries a confidence and a plain-English reason. Rules were written from
  the dev phrasings and general FOI knowledge only, never from test records.
* CLASSIFIER: a sentence classifier (TF-IDF words + character n-grams,
  logistic regression) trained on dev records. It generalises to wording
  the rules do not cover, and its strongest terms are shown as the reason.

Confidence per sentence and category = 1 - (1 - rule) * (1 - classifier),
so either detector can raise a flag and agreement raises it further.

Routing (who must look at it):
  * confidence >= threshold                     -> officer review
  * high-harm category and confidence >= floor  -> officer review
  * anything else above 0.05 is shown as a low-confidence hint
Nothing is ever redacted or released by this module.
"""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .exemptions import CATEGORIES, CATEGORY_IDS, HIGH_HARM

DEFAULT_THRESHOLD = 0.5
DEFAULT_FLOOR = 0.25
HINT_MIN = 0.05

# (category, pattern, confidence, reason)
RULES = [
    # s56 privacy
    ("s56_privacy", r"\bDOB\b:? ?\d{1,2}/\d{1,2}/\d{2,4}", .95, "Date of birth"),
    ("s56_privacy", r"\b(born on|date of birth)\b", .80, "Date of birth wording"),
    ("s56_privacy", r"[\w.+-]+@[\w-]+\.[\w.]+", .90, "Email address"),
    ("s56_privacy", r"\b04\d{2} ?\d{3} ?\d{3}\b", .90, "Mobile number"),
    ("s56_privacy", r"\b08 ?\d{4} ?\d{4}\b", .55, "Landline number"),
    ("s56_privacy", r"\b\d{1,4}[A-Za-z]? [A-Z][a-z]+(?: [A-Z][a-z]+)? (Street|St|Road|Rd|Avenue|Ave|Drive|Crescent|Court|Place|Lane|Way|Circuit|Parade|Terrace|Close|Grove|Boulevard)\b", .85, "Street address"),
    ("s56_privacy", r"\b(client|tenancy|customer|file) (number|no)\b|\b[A-Z]{2}-\d{6}\b", .85, "Client or file number"),
    ("s56_privacy", r"\b(tenant|applicant|occupant|client|leaseholder|complainant|resident),? (?:named )?[A-Z][a-z]+ [A-Z][a-z]+", .80, "Named member of the public"),
    ("s56_privacy", r"\b(evicted|rent arrears|lodged a complaint)\b", .55, "Personal circumstances"),
    ("s56_privacy", r"\b(behind on rent|aged \d{1,3}\b)", .60, "Personal circumstances"),
    # s56 health
    ("s56_health", r"\b(dialysis|diagnosed|disability|mental health|pregnan\w*|medication|heart condition|medical certificates?)\b", .85, "Health condition or treatment"),
    ("s56_health", r"\b(hospital|admitted to|wheelchair|renal|GP)\b", .50, "Health service reference"),
    ("s56_health", r"\b(diabetes|psychiatric|illness|surgery|disease|asthma|chemotherapy|treatment days)\b", .85, "Medical condition or procedure"),
    # s56 cultural
    ("s56_cultural", r"\b(sacred site|men's business|women's (site|business)|sorry business|ceremony ground|dreaming track)\b", .90, "Sacred or restricted cultural matter"),
    ("s56_cultural", r"\b(restricted to initiated|initiat\w+|AAPA)\b", .80, "Restricted cultural knowledge"),
    ("s56_cultural", r"\bdeceased\b", .70, "Reference to a deceased person"),
    ("s56_cultural", r"\b(late (leader|elder)|ceremonial)\b", .75, "Cultural protocol"),
    ("s56_cultural", r"\bout of respect\b", .50, "Cultural protocol wording"),
    ("s56_cultural", r"\b(custodians?|traditional owners|Land Council)\b", .45, "Traditional owner reference"),
    # s45 cabinet
    ("s45_cabinet", r"\bcabinet[- ]in[- ]confidence\b", .97, "Cabinet security marking"),
    ("s45_cabinet", r"\b(cabinet (submission|paper|decision|minute)|prepared for cabinet|cabinet's consideration|cabinet deliberated)\b", .88, "Cabinet document"),
    ("s45_cabinet", r"\bexecutive council\b", .75, "Executive Council"),
    ("s45_cabinet", r"\b(budget review subcommittee|expenditure review committee)\b", .85, "Cabinet committee"),
    ("s45_cabinet", r"\b(in|to) cabinet\b|\bministers (discussed|are yet to agree)\b", .70, "Cabinet process"),
    ("s45_cabinet", r"\bcabinet\b", .40, "Mentions Cabinet"),
    # s46 law enforcement
    ("s46_law_enforcement", r"\b(active investigation|INV-\d{4}-\w+)\b", .92, "Active investigation"),
    ("s46_law_enforcement", r"\b(informant|tip-off)\b", .88, "Informant"),
    ("s46_law_enforcement", r"\b(alarm codes?|access codes?|blind spot)\b", .85, "Security weakness"),
    ("s46_law_enforcement", r"\b(police intelligence|surveillance|ongoing operation|sly grog\w*)\b", .75, "Enforcement operations"),
    ("s46_law_enforcement", r"\bsuspect\b", .55, "Suspect"),
    ("s46_law_enforcement", r"\b(warrant|covert|confidential source|witness)\b", .80, "Enforcement method or source"),
    ("s46_law_enforcement", r"\bcharges\b", .55, "Charges"),
    # s49(d) legal privilege
    ("s49d_legal_privilege", r"\b(legal professional privilege|legally privileged|privileged and confidential)\b", .96, "Privilege marking"),
    ("s49d_legal_privilege", r"\b(solicitor for the northern territory|SFNT)\b", .85, "Government lawyer"),
    ("s49d_legal_privilege", r"\b(legal advice|counsel's opinion|counsel)\b", .78, "Legal advice"),
    ("s49d_legal_privilege", r"\b(liable|enforceable|prospects of)\b", .45, "Legal assessment"),
    ("s49d_legal_privilege", r"\b(lawyers?|barrister'?s?|crown counsel)\b", .78, "Lawyer's view"),
    # s52 deliberative
    ("s52_deliberative", r"\b(pre-decisional|for discussion only|working draft|internal consultation|final position)\b", .85, "Deliberative marking"),
    ("s52_deliberative", r"\b(preferred (approach|option)|not been endorsed|canvassed|weighing whether|possible option)\b", .70, "Options under consideration"),
    ("s52_deliberative", r"\bdraft\b", .35, "Draft"),
    ("s52_deliberative", r"\b(my (personal|preliminary) view|not yet settled|none has been chosen|thinking out loud|still weighing|not made a recommendation)\b", .75, "Unsettled view"),
    # s57 commercial
    ("s57_commercial", r"\b(commercial[- ]in[- ]confidence|trade secret|in confidence)\b", .90, "Confidentiality marking"),
    ("s57_commercial", r"\b(tender pricing|pricing breakdown|schedule of rates|unit rate|profit margin|evaluation scores)\b", .82, "Pricing or evaluation detail"),
    ("s57_commercial", r"\bscored \d+ out of\b", .75, "Tenderer score"),
    ("s57_commercial", r"\b(bid financials|financial statements|insurance premiums|wage costs|hourly (rates|labour charge)|discount)\b", .80, "Business financial detail"),
    ("s57_commercial", r"\$\d[\d,]*", .30, "Dollar figure"),
]
# Domain pack for real business correspondence. Written from the FIRST 40
# labelled Enron emails only (data/real/enron_labels_sample.jsonl); the other
# 40 are the held-out check. Enabled per collection, so NT results are unchanged.
PACKS = {
    "correspondence": [
        ("s56_privacy", r"\b(cell phone|home fax|home phone|pager)\b", .80, "Personal contact detail"),
        ("s56_privacy", r"\(?\b\d{3}\)?[- .]\d{3}-\d{4}\b", .60, "Phone number"),
        ("s56_privacy", r"@(aol|hotmail|yahoo|gmail)\.", .75, "Personal email account"),
        ("s56_privacy", r"\b(resume|summer intern|interview|offer letter|vacation|my family|anniversary|credit card)\b", .60, "Personal or employment matter"),
        ("s57_commercial", r"\b(pricing|price|deal ?#?\s?\d+|nomination|counterpart(y|ies)|gain to|profit)\b", .55, "Deal or pricing detail"),
        ("s57_commercial", r"\b\d+(,\d{3})* ?/ ?d\b|\bmmbtu", .50, "Volumes"),
        ("s49d_legal_privilege", r"\b(non-disclosure|intellectual property|attorneys?|legal\))", .60, "Legal matter"),
        ("s52_deliberative", r"\b(review (my|these) conclusions|before I send|revised draft|for your comments|evaluation of)\b", .60, "Draft or evaluation"),
    ],
}

COMPILED = [(c, re.compile(p, re.IGNORECASE if not re.search(r"\[A-Z\]", p) else 0), conf, why)
            for c, p, conf, why in RULES]

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\d$])")


def split_sentences(text):
    """Sentences with character offsets."""
    out, pos = [], 0
    for part in _SPLIT.split(text):
        start = text.index(part, pos)
        out.append((start, start + len(part), part))
        pos = start + len(part)
    return out


def _compile(rules):
    return [(c, re.compile(p, re.IGNORECASE if not re.search(r"\[A-Z\]", p) else 0), conf, why)
            for c, p, conf, why in rules]


COMPILED_PACKS = {k: _compile(v) for k, v in PACKS.items()}


def rule_scores(sentence, packs=()):
    scores, reasons = {}, {}
    rules = COMPILED + [r for k in packs for r in COMPILED_PACKS[k]]
    for cat, rx, conf, why in rules:
        m = rx.search(sentence)
        if m and conf > scores.get(cat, 0):
            scores[cat] = conf
            reasons[cat] = f'Rule: {why} ("{m.group(0)}")'
    return scores, reasons


# ---------------------------------------------------------------------------
# Optional language model (spaCy en_core_web_md): person names and word
# vectors. Everything works without it; the build reports which mode ran.
# ---------------------------------------------------------------------------
_NLP = None
_CACHE = {}
NAME_CONF = 0.4          # chosen by leave-phrasing-out CV on dev (experiments/detector_cv.py)
VEC_WEIGHT = 8.0


def language_model():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_md", exclude=["parser", "lemmatizer", "tagger", "attribute_ruler", "senter"])
        except Exception:
            _NLP = False
    return _NLP or None


def analyse(sents):
    """(person names, unit word vector) per sentence, cached by text."""
    nlp = language_model()
    todo = [t for t in dict.fromkeys(sents) if t not in _CACHE]
    if todo and nlp:
        for t, doc in zip(todo, nlp.pipe(todo, batch_size=256)):
            v = doc.vector.astype(np.float32)
            n = np.linalg.norm(v)
            _CACHE[t] = ([e.text for e in doc.ents if e.label_ == "PERSON"], v / n if n else v)
    return [_CACHE.get(t, ([], None)) for t in sents]


def _ner_rule(sent, persons, record, conf=None):
    """A person named in the body is a privacy candidate. Staff writing or
    receiving the record, and bare greetings and sign-offs, are not."""
    if len(sent.split()) < 5:
        return 0.0, None
    skip = " ".join(str(record.get(k, "")) for k in ("from", "to", "author")).lower()
    names = [p for p in persons if len(p) > 3 and p.lower() not in skip and p.split()[0].lower() not in skip.split()]
    if not names:
        return 0.0, None
    return (NAME_CONF if conf is None else conf), f'Names a person ("{names[0]}")'


class SensitivityDetector:
    def __init__(self, use_ner=False, use_vectors=False):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        self.clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
        self.classes = None
        lm = (use_ner or use_vectors) and language_model() is not None
        self.use_ner, self.use_vectors = bool(use_ner and lm), bool(use_vectors and lm)

    @property
    def mode(self):
        return "rules + classifier" + (" + names" if self.use_ner else "") + (" + word vectors" if self.use_vectors else "")

    # ------------------------------------------------------------------
    def fit(self, records, gt_by_id, split="dev", exclude_spans=None):
        """Train on sentences from records in the given split only.
        exclude_spans: {(record_id, span_start)} whose sentences are left out
        (used by cross-validation to hold out whole phrasings)."""
        X, y = [], []
        exclude_spans = exclude_spans or set()
        for r in records:
            g = gt_by_id[r["record_id"]]
            if g["split"] != split:
                continue
            gold = [s for s in g["spans"] if not s["decoy"]]
            for a, b, sent in split_sentences(r["body"]):
                lab, skip = "none", False
                for s in gold:
                    if a < s["end"] and s["start"] < b:
                        lab = s["category"]
                        skip |= (r["record_id"], s["start"]) in exclude_spans
                if skip:
                    continue
                X.append(sent)
                y.append(lab)
        self.word.fit(X)
        self.char.fit(X)
        self.clf.fit(self._vec(X), y)
        self.classes = list(self.clf.classes_)
        self._word_names = np.array(self.word.get_feature_names_out())
        self.n_train = len(X)
        return self

    def _vec(self, sents):
        parts = [self.word.transform(sents), self.char.transform(sents)]
        if self.use_vectors:
            parts.append(csr_matrix(np.vstack([v if v is not None else np.zeros(300, np.float32) for _, v in analyse(sents)]) * VEC_WEIGHT))
        return hstack(parts).tocsr()

    def _top_terms(self, sent, cat, k=3):
        ci = self.classes.index(cat)
        wv = self.word.transform([sent])
        coefs = self.clf.coef_[ci][: wv.shape[1]]
        contrib = wv.multiply(coefs).toarray().ravel()
        idx = [i for i in np.argsort(-contrib)[:k] if contrib[i] > 0]
        return [self._word_names[i] for i in idx]

    # ------------------------------------------------------------------
    def detect(self, record, packs=(), use_rules=True, names=None):
        """All candidate flags for one record, each with a confidence and
        reasons. Routing is applied later so the threshold can change live."""
        sents = split_sentences(record["body"])
        texts = [s[2] for s in sents]
        probs = self.clf.predict_proba(self._vec(texts))
        people = [x[0] for x in analyse(texts)] if (self.use_ner if names is None else names and self.use_ner) else [[] for _ in texts]
        flags = []
        for (a, b, sent), p, persons in zip(sents, probs, people):
            rs, rr = rule_scores(sent, packs) if use_rules else ({}, {})
            nc, nwhy = _ner_rule(sent, persons, record)
            if nc > rs.get("s56_privacy", 0):
                rs["s56_privacy"], rr["s56_privacy"] = nc, nwhy
            cand = []
            for cat in CATEGORY_IDS:
                ml = float(p[self.classes.index(cat)]) if cat in self.classes else 0.0
                rule = rs.get(cat, 0.0)
                conf = 1 - (1 - rule) * (1 - ml)
                if conf >= HINT_MIN:
                    reasons = []
                    if rule:
                        reasons.append(rr[cat])
                    if ml >= 0.1:
                        terms = self._top_terms(sent, cat)
                        reasons.append(f"Classifier {ml:.2f}" + (f": {', '.join(terms)}" if terms else ""))
                    cand.append((conf, cat, rule, ml, reasons))
            cand.sort(reverse=True)
            for conf, cat, rule, ml, reasons in cand[:2]:  # at most two categories per sentence
                flags.append({
                    "flag_id": f"{record['record_id']}:{len(flags)}",
                    "start": a, "end": b, "text": sent,
                    "category": cat, "label": CATEGORIES[cat]["label"], "section": CATEGORIES[cat]["section"],
                    "confidence": round(conf, 3), "rule_conf": round(rule, 3), "ml_conf": round(ml, 3),
                    "high_harm": cat in HIGH_HARM, "reasons": reasons,
                })
        return flags


def is_routed(flag, threshold=DEFAULT_THRESHOLD, floor=DEFAULT_FLOOR, use_floor=True):
    if flag["confidence"] >= threshold:
        return True
    return use_floor and flag["high_harm"] and flag["confidence"] >= floor

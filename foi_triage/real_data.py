"""Real data sources, prepared for the pipeline.

1. ENRON EMAILS (real correspondence)
   Source: Enron-Spam dataset (Metsis, Androutsopoulos & Paliouras, 2006),
   CSV release github.com/MWiechmann/enron_spam_data. We use only the "ham"
   messages: 16,545 real emails from Enron employees, released publicly
   during the US FERC investigation. That release is lower-cased and
   tokenised ("re : issue"), so we undo the tokenisation for readability.

2. SARA QUERIES (real information needs)
   Source: McKechnie, McDonald & Macdonald, "A Sensitivity-Aware Test
   Collection for Search Among Personal Information", SIGIR 2026.
   github.com/JackMcKechnie/SARA-A-Collection-of-Sensitivity-Aware-Relevance-Assessments
   50 information needs about Enron, each with 3 crowdsourced queries.
   CC BY-NC 4.0. They are used as access requests against the Enron emails.
   (SARA's relevance labels refer to the Berkeley-labelled subset whose
   text is hosted on Zenodo, which this build environment could not reach,
   so retrieval on Enron is shown qualitatively, not scored.)

3. NT ACCESS REQUESTS (real requests to NT bodies)
   Source: righttoknow.org.au, public request pages for 8 NT bodies.
   43 request titles with date and outcome. Used to test the request parser
   on real wording and to show real outcomes in the request register.
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"


def detokenize(text):
    t = text.replace("\r", "")
    t = re.sub(r"(?:- ){4,}-?", "----- ", t)                   # "- - - - -" rules
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = re.sub(r"(\d) / (\d)", r"\1/\2", t)                      # 12 / 14 / 99
    t = re.sub(r"(\d) / (\d)", r"\1/\2", t)
    t = re.sub(r"(\d) : (\d)", r"\1:\2", t)                      # 10 : 18
    t = re.sub(r" ([,.;:?!%)])", r"\1", t)
    t = re.sub(r"([($]) ", r"\1", t)
    t = re.sub(r" ' (s|t|re|ve|ll|d|m)\b", r"'\1", t)
    t = re.sub(r" n't\b", "n't", t)
    t = re.sub(r" @ ", "@", t)
    t = re.sub(r"(\w) - (\w)", r"\1-\2", t)
    t = re.sub(r"\bi\b", "I", t)
    # sentence-case: capitalise after sentence ends and line starts
    t = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def load_enron(n=1500, seed=7, csv_path=None):
    import pandas as pd
    csv_path = csv_path or REAL / "enron_spam_data.csv"
    df = pd.read_csv(csv_path)
    ham = df[df["Spam/Ham"] == "ham"].dropna(subset=["Message"])
    ham = ham[ham["Message"].str.len().between(250, 2600)]
    ham = ham.sample(n=n, random_state=seed)
    out = []
    for _, row in ham.iterrows():
        body = detokenize(str(row["Message"]))
        subj = detokenize(str(row["Subject"])) if isinstance(row["Subject"], str) else "(no subject)"
        out.append({
            "record_id": f"ENR-{int(row['Message ID']):05d}",
            "agency": "Enron Corp (real email)", "cc": [], "doc_type": "Email",
            "date": str(row["Date"])[:10], "from": "", "to": "",
            "subject": subj[:120], "body": body,
        })
    return out


def load_sara_requests():
    needs = {}
    for line in (REAL / "information_needs.tsv").read_text().splitlines():
        if "\t" in line:
            i, txt = line.split("\t", 1)
            needs[int(i)] = txt.strip()
    queries = {}
    for line in (REAL / "repeated_queries.tsv").read_text().splitlines():
        if "\t" in line:
            i, q = line.split("\t", 1)
            queries[int(i)] = q.strip()
    reqs = []
    for i, need in sorted(needs.items()):
        qs = [queries.get(3 * i - k) for k in (2, 1, 0)]
        reqs.append({"request_id": f"SARA-{i:02d}", "request_text": qs[0] or need,
                     "information_need": need, "alt_queries": [q for q in qs[1:] if q],
                     "received": "2026-09-21"})
    return reqs


def load_nt_requests():
    return json.loads((REAL / "nt_righttoknow_requests.json").read_text())


def prepare(n_enron=1500):
    recs = load_enron(n_enron)
    with open(REAL / "enron_records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    reqs = load_sara_requests()
    with open(REAL / "sara_requests.jsonl", "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    return recs, reqs


if __name__ == "__main__":
    r, q = prepare()
    print(len(r), "Enron emails,", len(q), "SARA requests")
    print(r[0]["subject"], "\n", r[0]["body"][:600])

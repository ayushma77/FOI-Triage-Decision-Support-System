"""Second annotator for the 80 real Enron emails.

The first labels (data/real/enron_labels_sample.jsonl) came from one
annotator. Before quoting the real-email results, a teammate labels the same
emails blind and we measure agreement (Cohen's kappa).

    python -m foi_triage.agreement --make-labeler        # writes outputs/label_emails.html
    # teammate opens it, labels all 80, clicks "Download CSV"
    python -m foi_triage.agreement second_labels.csv     # agreement report

Kappa reading (Landis & Koch, 1977): <0.2 slight, 0.2-0.4 fair, 0.4-0.6
moderate, 0.6-0.8 substantial, >0.8 almost perfect.
"""

import csv
import html
import json
import sys
from pathlib import Path

from .exemptions import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"


def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def cohen_kappa(a, b):
    """Kappa for two lists of binary labels."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def read_second(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("sensitive") or "").strip().lower()
            if v in ("yes", "no", "true", "false", "1", "0"):
                out[row["record_id"]] = {"sensitive": v in ("yes", "true", "1"),
                                         "categories": [c for c in (row.get("categories") or "").split(";") if c]}
    return out


def compare(first, second):
    ids = [i for i in first if i in second]
    a = [first[i]["sensitive"] for i in ids]
    b = [second[i]["sensitive"] for i in ids]
    both = [i for i in ids if first[i]["sensitive"] and second[i]["sensitive"]]
    cat_agree = sum(bool(set(first[i]["categories"]) & set(second[i]["categories"])) for i in both)
    split = {"first 40 (rule source)": ids[:40], "last 40 (held out)": ids[40:]}
    return {
        "compared": len(ids),
        "agreement": round(sum(x == y for x, y in zip(a, b)) / len(ids), 3) if ids else None,
        "kappa": round(cohen_kappa(a, b), 3) if ids else None,
        "kappa_by_split": {k: round(cohen_kappa([first[i]["sensitive"] for i in v], [second[i]["sensitive"] for i in v]), 3)
                           for k, v in split.items() if v},
        "both_sensitive": len(both), "same_category_when_both_sensitive": cat_agree,
        "first_only": [i for i in ids if first[i]["sensitive"] and not second[i]["sensitive"]],
        "second_only": [i for i in ids if second[i]["sensitive"] and not first[i]["sensitive"]],
    }


def make_labeler(out=ROOT / "outputs" / "label_emails.html"):
    labels = _jsonl(REAL / "enron_labels_sample.jsonl")
    recs = {r["record_id"]: r for r in _jsonl(REAL / "enron_records.jsonl")}
    items = [{"id": x["record_id"], "sub": recs[x["record_id"]]["subject"], "date": recs[x["record_id"]]["date"],
              "body": recs[x["record_id"]]["body"][:2500]} for x in labels]   # first labels deliberately NOT included
    cats = {k: v["label"] for k, v in CATEGORIES.items()}
    page = LABELER.replace("/*ITEMS*/[]", json.dumps(items)).replace("/*CATS*/{}", json.dumps(cats))
    Path(out).write_text(page)
    return out


LABELER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Email labelling</title><style>
[hidden]{display:none!important}
:root{--bg:#f4f5f8;--card:#fff;--ink:#1d2330;--muted:#5f6878;--accent:#2f5bd3;--line:#dfe3ea}
@media (prefers-color-scheme:dark){:root{--bg:#171a21;--card:#20242d;--ink:#e8ebf2;--muted:#9aa3b5;--accent:#7d9bff;--line:#323846}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:820px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:20px;margin:0 0 4px}.note{color:var(--muted);font-size:13.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-top:16px}
pre{white-space:pre-wrap;font:14px/1.6 Georgia,serif;max-height:46vh;overflow:auto;margin:10px 0 0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{font:inherit;font-weight:600;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:10px;padding:8px 14px;cursor:pointer}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
label.cat{display:inline-flex;gap:6px;align-items:center;font-size:13px;border:1px solid var(--line);border-radius:999px;padding:4px 10px}
.bar{height:6px;border-radius:3px;background:var(--line);margin-top:10px}.bar i{display:block;height:100%;border-radius:3px;background:var(--accent)}
</style></head><body><main>
<h1>Label 80 real emails</h1>
<p class="note">For each email: would an FOI officer need to review it before release? Yes if it contains personal details, health, legal advice,
commercial pricing or deal terms, drafts or deliberations, or anything else that might be exempt. Label independently: don't look at anyone else's labels.
Progress is saved in this browser. When done, click Download CSV and send the file back.</p>
<div class="card" id="card"></div>
<div class="row"><button id="prev">Previous</button><button id="next">Next</button><span class="note" id="count"></span><span style="flex:1"></span><button id="dl">Download CSV</button></div>
<div class="row"><label class="note">Your name <input id="who" style="font:inherit;padding:4px 8px"></label></div>
</main><script>
const ITEMS = /*ITEMS*/[], CATS = /*CATS*/{};
let L = {}, i = 0; try { L = JSON.parse(localStorage.getItem("foi-labels") || "{}"); } catch(e){}
const save = () => { try { localStorage.setItem("foi-labels", JSON.stringify(L)); } catch(e){} };
const esc = t => String(t).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function draw(){ const it = ITEMS[i], l = L[it.id] || {cats:[]}, done = ITEMS.filter(x => L[x.id] && L[x.id].s != null).length;
  document.getElementById("card").innerHTML = `<div class="note">${i+1} of ${ITEMS.length} · ${esc(it.id)} · ${esc(it.date)}</div><b>${esc(it.sub)}</b><pre>${esc(it.body)}</pre>
  <div class="row"><b>Needs review?</b><button data-s="1" class="${l.s===true?"on":""}">Yes</button><button data-s="0" class="${l.s===false?"on":""}">No</button></div>
  <div class="row" ${l.s?"":"hidden"}>${Object.entries(CATS).map(([k,v]) => `<label class="cat"><input type="checkbox" data-c="${k}" ${l.cats.includes(k)?"checked":""}>${esc(v)}</label>`).join("")}</div>
  <div class="bar"><i style="width:${done/ITEMS.length*100}%"></i></div>`;
  document.getElementById("count").textContent = `${done} labelled`; }
document.addEventListener("click", e => { const it = ITEMS[i];
  if (e.target.dataset.s != null){ L[it.id] = {...(L[it.id] || {cats:[]}), s: e.target.dataset.s === "1"}; save(); draw(); }
  if (e.target.id === "next"){ i = Math.min(ITEMS.length - 1, i + 1); draw(); }
  if (e.target.id === "prev"){ i = Math.max(0, i - 1); draw(); }
  if (e.target.id === "dl"){ const who = document.getElementById("who").value || "second annotator";
    const rows = [["record_id","sensitive","categories","annotator"], ...ITEMS.map(x => [x.id, L[x.id]?.s == null ? "" : L[x.id].s ? "yes" : "no", (L[x.id]?.cats || []).join(";"), who])];
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.map(r => r.map(v => /[",\\n]/.test(v) ? `"${v.replace(/"/g,'""')}"` : v).join(",")).join("\\n")], {type:"text/csv"}));
    a.download = "second_labels.csv"; a.click(); } });
document.addEventListener("change", e => { const c = e.target.dataset.c; if (!c) return; const l = L[ITEMS[i].id]; l.cats = e.target.checked ? [...new Set([...l.cats, c])] : l.cats.filter(x => x !== c); save(); });
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (e.key === "y") document.querySelector('[data-s="1"]').click(); if (e.key === "n") document.querySelector('[data-s="0"]').click();
  if (e.key === "ArrowRight") document.getElementById("next").click(); if (e.key === "ArrowLeft") document.getElementById("prev").click(); });
draw();
</script></body></html>
"""


def main(argv):
    if argv and argv[0] == "--make-labeler":
        print("Wrote", make_labeler())
        return None
    if not argv:
        print(__doc__)
        return None
    first = {x["record_id"]: x for x in _jsonl(REAL / "enron_labels_sample.jsonl")}
    first = {k: first[k] for k in first}                 # keep file order (first 40 dev, last 40 held out)
    r = compare(first, read_second(argv[0]))
    print(json.dumps(r, indent=1))
    (ROOT / "outputs" / "annotator_agreement.json").write_text(json.dumps(r, indent=1))
    return r


if __name__ == "__main__":
    main(sys.argv[1:])

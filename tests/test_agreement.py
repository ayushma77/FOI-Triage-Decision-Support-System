"""Second-annotator agreement (Cohen's kappa)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foi_triage.agreement import cohen_kappa, compare, make_labeler, read_second  # noqa: E402


def test_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) - 0.0) < 1e-9
    # 20 items: 8 both yes, 8 both no, 4 disagree → po .8, pe .5 → .6
    a = [1] * 8 + [0] * 8 + [1, 1, 0, 0]
    b = [1] * 8 + [0] * 8 + [0, 0, 1, 1]
    assert abs(cohen_kappa(a, b) - 0.6) < 1e-9


def test_compare_reads_csv_and_lists_disagreements(tmp_path):
    first = {"E1": {"sensitive": True, "categories": ["s56_privacy"]}, "E2": {"sensitive": False, "categories": []},
             "E3": {"sensitive": True, "categories": ["s57_commercial"]}}
    p = tmp_path / "second.csv"
    p.write_text("record_id,sensitive,categories,annotator\nE1,yes,s56_privacy,Sam\nE2,yes,s52_deliberative,Sam\nE3,no,,Sam\nE4,,,Sam\n")
    r = compare(first, read_second(p))
    assert r["compared"] == 3 and r["first_only"] == ["E3"] and r["second_only"] == ["E2"]
    assert r["same_category_when_both_sensitive"] == 1


def test_labeler_page_is_blind(tmp_path):
    out = make_labeler(tmp_path / "l.html")
    t = out.read_text()
    assert "ENR-" in t and '"sensitive": true' not in t and '"categories": ["' not in t and "Download CSV" in t

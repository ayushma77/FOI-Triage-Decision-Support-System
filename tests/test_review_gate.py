"""The human gate is the core promise, so it is tested directly."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foi_triage.review import ReleaseBlocked, ReviewSession  # noqa: E402
from foi_triage.sensitivity import is_routed  # noqa: E402


def packet():
    flag = {"flag_id": "R1:0", "routed": True, "label": "Personal information", "text": "Jo Smith, 0412 345 678",
            "category": "s56_privacy", "confidence": 0.9, "high_harm": False}
    hint = {"flag_id": "R1:1", "routed": False, "label": "Commercial", "text": "$300", "category": "s57_commercial",
            "confidence": 0.3, "high_harm": False}
    return {"request_id": "REQ-T", "records": [
        {"record_id": "R1", "flags": [flag, hint], "has_routed_flags": True},
        {"record_id": "R2", "flags": [], "has_routed_flags": False},
    ]}


def test_finalise_blocked_until_everything_decided():
    s = ReviewSession(packet(), "Officer A")
    with pytest.raises(ReleaseBlocked) as e:
        s.finalise()
    assert "flag R1:0" in e.value.outstanding
    assert "record R2" in e.value.outstanding  # unflagged records still need an officer


def test_unflagged_record_still_needs_outcome():
    s = ReviewSession(packet(), "Officer A")
    s.decide_flag("R1:0", "redact")
    s.set_record_outcome("R1", "release_with_redactions")
    with pytest.raises(ReleaseBlocked):
        s.finalise()
    s.set_record_outcome("R2", "release_in_full")
    result = s.finalise()
    assert result["redactions"] == ["R1:0"]


def test_record_outcome_blocked_while_flag_open():
    s = ReviewSession(packet(), "Officer A")
    with pytest.raises(ReleaseBlocked):
        s.set_record_outcome("R1", "release_in_full")


def test_release_of_flagged_passage_needs_reason():
    s = ReviewSession(packet(), "Officer A")
    with pytest.raises(ValueError):
        s.decide_flag("R1:0", "release")
    s.decide_flag("R1:0", "release", reason="Applicant's own details")
    assert s.flag_decisions["R1:0"]["reason"]


def test_cannot_release_in_full_after_redaction():
    s = ReviewSession(packet(), "Officer A")
    s.decide_flag("R1:0", "redact")
    with pytest.raises(ValueError):
        s.set_record_outcome("R1", "release_in_full")


def test_session_requires_named_officer_and_logs_actions():
    with pytest.raises(ValueError):
        ReviewSession(packet(), "")
    s = ReviewSession(packet(), "Officer A")
    s.decide_flag("R1:0", "consult")
    assert [a["action"] for a in s.audit] == ["session_opened", "flag_consult"]
    assert all(a["officer"] == "Officer A" for a in s.audit)


def test_high_harm_floor_routes_low_confidence_flags():
    f = {"confidence": 0.3, "high_harm": True}
    assert is_routed(f, threshold=0.5, floor=0.25)
    assert not is_routed(f, threshold=0.5, floor=0.25, use_floor=False)
    assert not is_routed({"confidence": 0.3, "high_harm": False}, threshold=0.5, floor=0.25)


def test_pipeline_never_fills_officer_decisions():
    from foi_triage.pipeline import build, load_data
    records, requests, gt_records, _ = load_data()
    packets, *_ = build(records, requests[:3], gt_records)
    for p in packets:
        for r in p["records"]:
            assert r["officer_outcome"] is None

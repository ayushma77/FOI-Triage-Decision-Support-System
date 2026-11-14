"""Multi-user workflow: roles, gates, separation of duties, and what the applicant can see."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foi_triage.workflow import USERS, Workflow, WorkflowError  # noqa: E402

CATS = {"s56_privacy": {"label": "Personal", "section": "s56 Privacy", "high_harm": False},
        "s56_health": {"label": "Health", "section": "s56 Privacy", "high_harm": True}}


class FakeRetriever:
    def search(self, text):
        return {"topic_terms": text.split()}, [{"record_id": "R1", "relevance": 1.0, "bm25": 2.0, "semantic": .5, "matched_terms": [], "reasons": []},
                                              {"record_id": "R2", "relevance": .5, "bm25": 1.0, "semantic": .4, "matched_terms": [], "reasons": []}]


@pytest.fixture
def wf(tmp_path):
    cols = {"nt": {
        "records": {"R1": {"subject": "One", "date": "2025-01-01", "body": "Call Jo on 0412 345 678 please."},
                    "R2": {"subject": "Two", "date": "2025-02-01", "body": "Routine update only."}},
        # flags: [i, category, confidence, start, end]
        "flags": {"R1": [[0, "s56_privacy", .9, 0, 31], [1, "s56_health", .3, 0, 4]], "R2": [[0, "s56_privacy", .2, 0, 7]]},
        "packets": {}, "retriever": FakeRetriever()}}
    return Workflow(tmp_path / "wf.json", cols, CATS, {"threshold": .5, "floor": .25})


def user(name):
    return {"username": name, **{k: v for k, v in USERS[name].items() if k != "password"}}


def lodge(wf):
    return wf.create_case(user("alex"), "All documents about repairs in Maningrida in 2025", "nt")["id"]


def complete(wf, cid):
    o = user("jordan")
    wf.act(cid, o, "decide", {"flag": "R1:0", "action": "redact"})
    wf.act(cid, o, "decide", {"flag": "R1:1", "action": "consult", "party": "Jo"})   # high-harm floor routes 0.3
    wf.act(cid, o, "consult_response", {"flag": "R1:1", "response": "objects", "note": "Medical detail"})
    wf.act(cid, o, "decide", {"flag": "R1:1", "action": "redact"})
    wf.act(cid, o, "outcome", {"record": "R1", "outcome": "release_with_redactions"})
    wf.act(cid, o, "outcome", {"record": "R2", "outcome": "release_in_full"})


def approve(wf, cid):
    for k in wf.case(cid)["review"]["sample"]:
        wf.act(cid, user("sam"), "spot_check", {"flag": k, "agree": True})
    return wf.act(cid, user("sam"), "approve", {})


def test_login_and_bad_password(wf):
    token, u = wf.login("sam", "demo")
    assert u["role"] == "decision_maker" and wf.user_for(token)["name"] == "Sam Ngo"
    with pytest.raises(WorkflowError):
        wf.login("sam", "nope")


def test_applicant_lodges_and_ai_triage_runs(wf):
    c = wf.case(lodge(wf))
    assert c["status"] == "in_review" and [r["id"] for r in c["records"]] == ["R1", "R2"]
    assert [e["action"] for e in c["events"]] == ["Request lodged", "AI triage complete"]


def test_applicant_cannot_act_and_never_sees_flags(wf):
    cid = lodge(wf)
    with pytest.raises(WorkflowError) as e:
        wf.act(cid, user("alex"), "decide", {"flag": "R1:0", "action": "release", "reason": "x"})
    assert e.value.code == 403
    v = wf.list_for(user("alex"))["cases"][0]
    assert "decisions" not in v and "records" not in v and v["public_status"] == "Being processed"


def test_submit_gate_blocks_until_complete(wf):
    cid = lodge(wf)
    with pytest.raises(WorkflowError) as e:
        wf.act(cid, user("jordan"), "submit", {})
    assert e.value.code == 409
    complete(wf, cid)
    assert wf.act(cid, user("jordan"), "submit", {})["status"] == "awaiting_approval"


def test_release_needs_reason_and_case_locks_after_submit(wf):
    cid = lodge(wf)
    with pytest.raises(WorkflowError):
        wf.act(cid, user("jordan"), "decide", {"flag": "R1:0", "action": "release"})
    complete(wf, cid)
    wf.act(cid, user("jordan"), "submit", {})
    with pytest.raises(WorkflowError) as e:
        wf.act(cid, user("jordan"), "reopen", {"record": "R1"})
    assert e.value.code == 409


def test_only_decision_maker_approves_and_return_needs_note(wf):
    cid = lodge(wf)
    complete(wf, cid)
    wf.act(cid, user("jordan"), "submit", {})
    with pytest.raises(WorkflowError) as e:
        wf.act(cid, user("jordan"), "approve", {})
    assert e.value.code == 403
    with pytest.raises(WorkflowError):
        wf.act(cid, user("sam"), "return", {"note": ""})
    c = wf.act(cid, user("sam"), "return", {"note": "Check R2 for names"})
    assert c["status"] == "returned" and c["note"] == "Check R2 for names"
    wf.act(cid, user("jordan"), "submit", {})
    assert approve(wf, cid)["status"] == "released"


def test_applicant_receives_redacted_package_and_notice(wf):
    cid = lodge(wf)
    complete(wf, cid)
    wf.act(cid, user("jordan"), "submit", {})
    approve(wf, cid)
    v = wf.view_for(user("alex"), wf.case(cid))
    r1 = next(d for d in v["package"] if d["id"] == "R1")
    assert any(p.get("redacted") == "s56" for p in r1["parts"])
    assert "0412 345 678" not in "".join(p.get("t", "") for p in r1["parts"])
    assert "Approved by Sam Ngo" in v["notice"] and "s56 Privacy" in v["notice"]


def test_state_persists_to_disk(wf, tmp_path):
    cid = lodge(wf)
    again = Workflow(tmp_path / "wf.json", wf.cols, CATS, {"threshold": .5, "floor": .25})
    assert again.case(cid)["text"].startswith("All documents")


def test_applicant_can_withdraw_own_request_only(wf):
    cid = lodge(wf)
    with pytest.raises(WorkflowError) as e:
        wf.act(cid, user("jordan"), "withdraw", {})
    assert e.value.code == 403
    assert wf.act(cid, user("alex"), "withdraw", {})["status"] == "withdrawn"
    with pytest.raises(WorkflowError):
        wf.act(cid, user("jordan"), "decide", {"flag": "R1:0", "action": "redact"})
    assert wf.view_for(user("alex"), wf.case(cid))["public_status"] == "Withdrawn"


def test_clarification_loop_pauses_and_resumes(wf):
    cid = lodge(wf)
    with pytest.raises(WorkflowError):
        wf.act(cid, user("alex"), "clarify", {"question": "Which years do you need?"})
    c = wf.act(cid, user("jordan"), "clarify", {"question": "Do you need contractor invoices as well as emails?"})
    assert c["status"] == "clarification"
    with pytest.raises(WorkflowError):          # can't send for approval while waiting on the applicant
        complete(wf, cid) or wf.act(cid, user("jordan"), "submit", {})
    v = wf.view_for(user("alex"), wf.case(cid))
    assert v["public_status"] == "Your reply is needed" and "invoices" in v["clarify"]["question"]
    c = wf.act(cid, user("alex"), "reply", {"answer": "Emails only, thanks."})
    assert c["status"] == "in_review" and c["clarify"]["answer"] == "Emails only, thanks."


def test_officer_can_flag_what_the_ai_missed_and_it_gates_and_redacts(wf):
    cid = lodge(wf)
    complete(wf, cid)
    c = wf.act(cid, user("jordan"), "manual_flag", {"record": "R2", "start": 0, "end": 7, "category": "s56_privacy"})
    key = c["manual"][0]["k"]
    assert key == "R2:m1" and f"flag {key}" in wf.outstanding(c)     # the new flag blocks sending until decided
    wf.act(cid, user("jordan"), "decide", {"flag": key, "action": "redact"})
    wf.act(cid, user("jordan"), "outcome", {"record": "R2", "outcome": "release_with_redactions"})
    wf.act(cid, user("jordan"), "submit", {})
    approve(wf, cid)
    r2 = next(d for d in wf.view_for(user("alex"), wf.case(cid))["package"] if d["id"] == "R2")
    assert "Routine" not in "".join(p.get("t", "") for p in r2["parts"])
    with pytest.raises(WorkflowError):
        wf.act(cid, user("alex"), "manual_flag", {"record": "R1", "start": 0, "end": 4, "category": "s56_privacy"})


def test_officer_flags_become_reviewable_rule_suggestions(wf):
    from foi_triage.feedback import as_rule_lines, suggest_pattern, suggestions
    import re
    pat = suggest_pattern("lock combination for the evidence room is 4471")
    assert re.search(pat, "The lock combination for the evidence room is 9902.", re.I)
    assert not re.search(pat, "The evidence room was repainted.", re.I)
    cid = lodge(wf)
    wf.act(cid, user("jordan"), "manual_flag", {"record": "R2", "start": 0, "end": 14, "category": "s56_privacy"})
    recs = {"nt": {k: {"body": v["body"]} for k, v in wf.cols["nt"]["records"].items()}}
    items = suggestions(wf.state, recs)
    assert items[0]["text"] == "Routine update" and items[0]["category"] == "s56_privacy"
    assert as_rule_lines(items)[0].lstrip().startswith('("s56_privacy", r"routine')


def test_consultation_must_be_answered_before_deciding_or_sending(wf):
    cid, o = lodge(wf), user("jordan")
    wf.act(cid, o, "decide", {"flag": "R1:0", "action": "redact"})
    c = wf.act(cid, o, "decide", {"flag": "R1:1", "action": "consult", "party": "Jo Smith"})
    assert c["consults"]["R1:1"]["party"] == "Jo Smith" and "consultation R1:1" in wf.outstanding(c)
    with pytest.raises(WorkflowError):          # can't finalise the passage until they've replied
        wf.act(cid, o, "decide", {"flag": "R1:1", "action": "release", "reason": "Public already"})
    with pytest.raises(WorkflowError):          # nor confirm the record
        wf.act(cid, o, "outcome", {"record": "R1", "outcome": "release_with_redactions"})
    wf.act(cid, o, "consult_response", {"flag": "R1:1", "response": "agrees"})
    wf.act(cid, o, "decide", {"flag": "R1:1", "action": "release", "reason": "Jo agreed to release"})
    wf.act(cid, o, "outcome", {"record": "R1", "outcome": "release_with_redactions"})
    wf.act(cid, o, "outcome", {"record": "R2", "outcome": "release_in_full"})
    assert wf.act(cid, o, "submit", {})["status"] == "awaiting_approval"
    assert "consulted 1 third party" in (approve(wf, cid)["notice"])


def test_decision_maker_must_spot_check_and_releases_are_sampled_first(wf):
    cid, o = lodge(wf), user("jordan")
    wf.act(cid, o, "decide", {"flag": "R1:0", "action": "release", "reason": "Applicant's own number"})
    wf.act(cid, o, "decide", {"flag": "R1:1", "action": "redact"})
    wf.act(cid, o, "outcome", {"record": "R1", "outcome": "release_with_redactions"})
    wf.act(cid, o, "outcome", {"record": "R2", "outcome": "release_in_full"})
    c = wf.act(cid, o, "submit", {})
    assert c["review"]["sample"][0] == "R1:0"          # the release is checked first
    with pytest.raises(WorkflowError) as e:
        wf.act(cid, user("sam"), "approve", {})
    assert e.value.code == 409
    with pytest.raises(WorkflowError):                  # disagreeing needs a reason
        wf.act(cid, user("sam"), "spot_check", {"flag": "R1:0", "agree": False})
    wf.act(cid, user("sam"), "spot_check", {"flag": "R1:0", "agree": False, "note": "That number is the neighbour's"})
    wf.act(cid, user("sam"), "spot_check", {"flag": "R1:1", "agree": True})
    with pytest.raises(WorkflowError):                  # a disagreement blocks approval
        wf.act(cid, user("sam"), "approve", {})
    assert wf.act(cid, user("sam"), "return", {"note": "See spot-check"})["status"] == "returned"


def test_hurried_uniform_review_is_flagged(wf):
    c = wf.case(lodge(wf))
    c["decisions"] = {f"R1:{i}": {"action": "redact", "reason": "", "by": "Jordan Lee", "time": f"2026-10-06T10:00:0{i}+00:00"} for i in range(9)}
    rv = wf.review_check(c)
    assert len(rv["warnings"]) == 2 and rv["fast"] == 8

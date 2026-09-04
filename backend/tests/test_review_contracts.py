import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.core.security import create_guest_access_token, create_one_time_token, decode_token, hash_one_time_token
from app.dependencies import GuestUser, get_current_guest
from app.models.schemas import CreateReviewRequest
from app.services.review_service import REPORT_HEADINGS, _coerce_node_result, _normalize_report, _validate_node_result
from app.runtime.langgraph_runtime import ReviewState, build_review_graph
from app.routers.reviews import _safe_filename, _validate_file_signature
from app.core.web_search import filter_relevant_results
from app.core.config import settings
from app.services.email_service import EmailDeliveryError, send_email
from app.services.guest_review_service import GuestReviewStore


def test_one_time_token_hash_round_trip():
    raw, hashed = create_one_time_token()
    assert raw
    assert hashed == hash_one_time_token(raw)


def test_guest_token_is_ephemeral_and_marked_as_guest():
    payload = decode_token(create_guest_access_token("guest:test"))
    assert payload is not None
    assert payload["sub"] == "guest:test"
    assert payload["guest"] is True


def test_guest_review_store_is_owner_scoped_and_not_persistent():
    store = GuestReviewStore()
    review = store.create("guest:one", "游客测试", 1)
    assert store.summary(review)["organization_id"] == "guest"
    assert store.get(review.id, "guest:one").id == review.id
    try:
        store.get(review.id, "guest:two")
    except KeyError:
        pass
    else:
        raise AssertionError("guest reviews must be isolated by owner token")
    store.purge("guest:one")
    assert review.id not in store.reviews


@pytest.mark.asyncio
async def test_guest_events_are_replayable_and_guest_dependency_avoids_database():
    store = GuestReviewStore()
    review = store.create("guest:one", "游客测试", 1)
    await store.emit(review, "session_status", {"status": "running"})
    assert review.events[0]["sequence"] == 1
    assert review.events[0]["session_id"] == review.id
    token = create_guest_access_token("guest:one")
    user = await get_current_guest(credentials=None, access_cookie=token)
    assert isinstance(user, GuestUser)
    assert user.id == "guest:one"


def test_guest_review_store_purges_expired_tasks(monkeypatch):
    store = GuestReviewStore()
    review = store.create("guest:one", "过期测试", 1)
    review.updated_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    monkeypatch.setattr(settings, "guest_review_ttl_minutes", 60)
    store.purge_expired()
    assert review.id not in store.reviews


def test_guest_review_store_limits_active_reviews(monkeypatch):
    store = GuestReviewStore()
    monkeypatch.setattr(settings, "guest_max_active_reviews", 1)
    store.create("guest:one", "第一个", 1)
    with pytest.raises(RuntimeError):
        store.create("guest:two", "第二个", 1)


def test_review_round_validation_and_report_headings():
    request = CreateReviewRequest(organization_id="org-1", topic="主题", max_round=5)
    assert request.max_round == 5
    report = _normalize_report("## 方案概述\n内容")
    for heading in REPORT_HEADINGS:
        assert heading in report
    assert report.index(REPORT_HEADINGS[0]) < report.index(REPORT_HEADINGS[-1])


def test_guest_report_normalization_preserves_topic_without_sql_session():
    report = _normalize_report("## 方案概述\n未命名评审", topic="评估客服自动化项目")
    assert "## 方案概述\n评估客服自动化项目" in report


def test_report_normalization_replaces_empty_model_sections_from_outputs_and_evidence():
    outputs = [
        {"agent_role": "benefit_argument", "content_markdown": "可降低人工处理成本"},
        {"agent_role": "risk_argument", "content_markdown": "实施需要额外预算"},
    ]
    evidence = [{"claim_text": "需确认预算", "verdict": "uncertain", "sources": []}]
    report = _normalize_report(
        "## 方案概述\n主题\n## 收益清单\n- 暂无\n## 风险与隐患清单\n- 暂无\n## 待确认不确定性点\n- 暂无\n## 参考证据来源列表\n- 暂无",
        topic="主题",
        outputs=outputs,
        evidence=evidence,
    )
    assert "- 可降低人工处理成本" in report
    assert "- 实施需要额外预算" in report
    assert "- 需确认预算" in report


def test_invalid_agent_output_is_rejected():
    try:
        _validate_node_result("benefit_argument", {"summary": "x", "claims": []})
    except ValueError:
        pass
    else:
        raise AssertionError("empty argument claims must fail validation")


def test_document_synthesis_output_is_coerced_to_summary():
    result = _coerce_node_result(
        "document_parse",
        {
            "topic": "主题",
            "synthesis": {"benefits": ["提升效率"], "risks": ["实施成本"]},
            "sources": [{"title": "公开资料"}],
        },
    )
    assert "summary" in result
    assert "提升效率" in result["summary"]


def test_topic_search_filters_unrelated_results_but_keeps_matching_sources():
    results = [
        {"title": "Mercedes-Benz 车型技术讨论", "body": "发动机和底盘参数", "url": "https://example.com/car"},
        {"title": "企业知识库升级实践", "body": "企业知识库检索与权限管理", "url": "https://example.com/kb"},
    ]
    filtered = filter_relevant_results("企业知识库升级", results)
    assert [item["url"] for item in filtered] == ["https://example.com/kb"]


def test_topic_search_without_relevant_sources_returns_empty():
    results = [{"title": "车型论坛", "body": "发动机参数讨论", "url": "https://example.com/car"}]
    assert filter_relevant_results("企业知识库升级", results) == []


def test_guest_claim_search_does_not_keep_unrelated_sources():
    results = [
        {"title": "Java 在线编译器", "body": "编程工具", "url": "https://example.com/java"},
        {"title": "企业知识库升级实践", "body": "知识检索效率与权限管理", "url": "https://example.com/kb"},
    ]
    assert [item["url"] for item in filter_relevant_results("企业知识库升级可提升检索效率", results)] == ["https://example.com/kb"]


def test_argument_claims_only_output_is_coerced_to_summary():
    result = _coerce_node_result(
        "benefit_argument",
        {"claims": ["降低人工处理成本", "缩短交付周期"]},
    )
    _validate_node_result("benefit_argument", result)
    assert result["summary"].startswith("收益论据：")


def test_summary_report_alias_is_coerced_to_markdown():
    result = _coerce_node_result("summary_report", {"report": "## 方案概述\n内容"})
    _validate_node_result("summary_report", result)
    assert result["markdown"].startswith("## 方案概述")


def test_langgraph_review_state_and_graph_compile():
    state: ReviewState = {"topic": "主题", "max_round": 1, "current_round": 0, "session_id": "s"}
    assert state["max_round"] == 1
    graph = build_review_graph()
    assert graph is not None


def test_upload_filename_is_confined_to_basename_and_signature_is_checked():
    assert _safe_filename("..\\..\\secret.pdf") == "secret.pdf"
    try:
        _validate_file_signature("document.pdf", b"not a pdf")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
    else:
        raise AssertionError("invalid PDF signature must be rejected")


def test_supabase_email_provider_does_not_silently_drop_invites(monkeypatch):
    monkeypatch.setattr(settings, "email_provider", "supabase")
    try:
        send_email("member@example.com", "邀请", "<p>邀请</p>")
    except EmailDeliveryError as exc:
        assert "EMAIL_PROVIDER" in str(exc)
    else:
        raise AssertionError("unsupported Supabase email provider must fail explicitly")

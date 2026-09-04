"""Ephemeral guest review runtime.

Guest sessions use the same DeepSeek-backed review stages as authenticated
reviews, but all task state, extracted text, outputs, evidence, and SSE replay
are kept in this process only. Nothing is written to SQL or object storage.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.core.config import settings
from app.core.sse_manager import sse_manager
from app.core.web_search import filter_relevant_results, search_web
from app.services.document_service import extract_document_text
from app.services.llm_service import structured
from app.services.review_service import (
    REPORT_HEADINGS,
    _classify_claim,
    _coerce_node_result,
    _normalize_report,
    _validate_node_result,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GuestReview:
    id: str
    owner_id: str
    topic: str | None
    max_round: int
    created_at: str
    updated_at: str
    current_round: int = 0
    current_stage: str = "draft"
    status: str = "draft"
    documents: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    report_markdown: str | None = None
    error_message: str | None = None
    sequence: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class GuestReviewStore:
    def __init__(self) -> None:
        self.reviews: dict[str, GuestReview] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def create(self, owner_id: str, topic: str | None, max_round: int) -> GuestReview:
        self.purge_expired()
        active = [item for item in self.reviews.values() if item.status in {"draft", "queued", "running", "awaiting_human"}]
        if len(active) >= settings.guest_max_active_reviews:
            raise RuntimeError("游客体验当前人数较多，请稍后重试")
        owner_reviews = [item for item in self.reviews.values() if item.owner_id == owner_id]
        if len(owner_reviews) >= settings.guest_max_reviews_per_session:
            oldest = min(owner_reviews, key=lambda item: item.updated_at)
            self._remove(oldest.id)
        review_id = f"guest-{uuid4()}"
        now = _now()
        review = GuestReview(review_id, owner_id, topic, max_round, now, now)
        self.reviews[review_id] = review
        return review

    def get(self, review_id: str, owner_id: str) -> GuestReview:
        self.purge_expired()
        review = self.reviews.get(review_id)
        if not review or review.owner_id != owner_id:
            raise KeyError(review_id)
        return review

    def purge(self, owner_id: str) -> None:
        for review_id, review in list(self.reviews.items()):
            if review.owner_id == owner_id:
                self._remove(review_id)

    def purge_expired(self) -> None:
        cutoff = time.time() - settings.guest_review_ttl_minutes * 60
        for review_id, review in list(self.reviews.items()):
            try:
                updated = datetime.fromisoformat(review.updated_at).timestamp()
            except ValueError:
                updated = 0
            if updated < cutoff:
                self._remove(review_id)

    def _remove(self, review_id: str) -> None:
        task = self.tasks.pop(review_id, None)
        if task and not task.done():
            task.cancel()
        self.reviews.pop(review_id, None)

    async def emit(self, review: GuestReview, event_type: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            review.sequence += 1
            review.updated_at = _now()
            data = {
                "session_id": review.id,
                "sequence": review.sequence,
                "timestamp": review.updated_at,
                "type": event_type,
                **payload,
            }
            review.events.append(data)
            if len(review.events) > 5000:
                del review.events[:-5000]
        await sse_manager.broadcast(review.id, event_type, data)

    async def add_output(self, review: GuestReview, agent: str, round_num: int, content: str, structured_data: dict[str, Any]) -> None:
        output = {
            "id": f"{review.id}-{agent}-{round_num}-{len(review.outputs) + 1}",
            "agent_role": agent,
            "round_num": round_num,
            "sequence": len(review.outputs) + 1,
            "content_markdown": content,
            "structured_data": structured_data,
            "created_at": _now(),
        }
        review.outputs.append(output)
        await self.emit(review, "agent_result", {"agent": agent, "round": round_num, "output_id": output["id"], "content": content, "structured_data": structured_data})

    @staticmethod
    def summary(review: GuestReview) -> dict[str, Any]:
        return {
            "id": review.id,
            "organization_id": "guest",
            "topic": review.topic,
            "max_round": review.max_round,
            "current_round": review.current_round,
            "current_stage": review.current_stage,
            "status": review.status,
            "document_count": len(review.documents),
            "evidence_count": len(review.evidence),
            "creator_id": review.owner_id,
            "creator_name": "游客体验",
            "created_at": review.created_at,
            "updated_at": review.updated_at,
        }

    def detail(self, review: GuestReview) -> dict[str, Any]:
        return {
            **self.summary(review),
            "documents": [
                {key: item[key] for key in ("id", "filename", "content_type", "size_bytes", "parse_status", "parse_error")}
                for item in review.documents
            ],
            "outputs": review.outputs,
            "report_markdown": review.report_markdown,
            "error_message": review.error_message,
        }

    async def run(self, review: GuestReview) -> None:
        async def stage(stage_name: str, status: str = "running", round_num: int | None = None) -> None:
            review.current_stage = stage_name
            review.status = status
            if round_num is not None:
                review.current_round = round_num
            await self.emit(review, "session_status", {"status": review.status, "stage": stage_name, "round": review.current_round})

        async def node(agent: str, round_num: int, messages: list[dict[str, str]], fallback: dict[str, Any]) -> dict[str, Any]:
            chunks: list[str] = []

            async def on_chunk(chunk: str) -> None:
                chunks.append(chunk)
                await self.emit(review, "agent_chunk", {"agent": agent, "round": round_num, "content": chunk})

            result = _coerce_node_result(agent, await structured(messages, fallback, on_chunk=on_chunk))
            _validate_node_result(agent, result)
            return result

        try:
            await stage("document_parse")
            await self.emit(review, "agent_start", {"agent": "document_parse", "round": 0})
            source = "\n\n".join(f"[{item['filename']}]\n{item.get('extracted_text', '')}" for item in review.documents)
            source = (f"主题：{review.topic}\n\n{source}" if review.topic else source).strip()
            document = await node(
                "document_parse",
                0,
                [{"role": "system", "content": "你是文档解析 Agent。只输出 JSON，包含 summary 和 claims。所有自然语言使用简体中文。"}, {"role": "user", "content": source[: settings.max_document_chars] or "请基于评审主题生成简短材料摘要。"}],
                {"summary": review.topic or "游客评审材料摘要", "claims": []},
            )
            await self.add_output(review, "document_parse", 0, document["summary"], document)

            for round_num in range(1, review.max_round + 1):
                await stage("benefit_argument", round_num=round_num)
                await self.emit(review, "agent_start", {"agent": "benefit_argument", "round": round_num})
                benefit = await node(
                    "benefit_argument",
                    round_num,
                    [{"role": "system", "content": "你是收益论证 Agent。只输出 JSON，包含 summary 和 claims 数组；使用简体中文，不得编造来源。"}, {"role": "user", "content": json.dumps({"topic": review.topic, "document": document, "round": round_num}, ensure_ascii=False)}],
                    {"summary": "从可行性、收益和有利条件分析该方案。", "claims": [{"claim": "该方案具备潜在业务收益，仍需结合真实材料验证。"}]},
                )
                await self.add_output(review, "benefit_argument", round_num, benefit["summary"], benefit)
                await self._fact_check(review, round_num, benefit, "benefit_argument")

                await stage("risk_argument", round_num=round_num)
                await self.emit(review, "agent_start", {"agent": "risk_argument", "round": round_num})
                risk = await node(
                    "risk_argument",
                    round_num,
                    [{"role": "system", "content": "你是风险研判 Agent。只输出 JSON，包含 summary 和 claims 数组；使用简体中文，不得编造来源。"}, {"role": "user", "content": json.dumps({"topic": review.topic, "document": document, "benefit": benefit, "round": round_num}, ensure_ascii=False)}],
                    {"summary": "识别落地约束、潜在风险和反面论据。", "claims": [{"claim": "方案落地依赖预算、资源和实施条件，当前材料不足以排除风险。"}]},
                )
                await self.add_output(review, "risk_argument", round_num, risk["summary"], risk)
                await self._fact_check(review, round_num, risk, "risk_argument")
                await self.emit(review, "round_complete", {"round": round_num, "max_round": review.max_round})

            await stage("summary_report", round_num=review.max_round)
            await self.emit(review, "agent_start", {"agent": "summary_report", "round": review.max_round})
            summary = await node(
                "summary_report",
                review.max_round,
                [{"role": "system", "content": "你是汇总评审 Agent。只输出 JSON，字段 markdown；必须包含五个固定标题，全部使用简体中文。"}, {"role": "user", "content": json.dumps({"topic": review.topic, "outputs": review.outputs, "evidence": review.evidence}, ensure_ascii=False)}],
                {"markdown": self._fallback_report(review)},
            )
            # Guest reviews do not have a SQL ReviewSession, so pass the topic
            # explicitly while normalizing the fixed report structure.
            review.report_markdown = _normalize_report(
                summary["markdown"],
                topic=review.topic,
                outputs=review.outputs,
                evidence=review.evidence,
            )
            await self.add_output(review, "summary_report", review.max_round, review.report_markdown, summary)
            review.current_stage = "completed"
            review.status = "completed"
            await self.emit(review, "report_ready", {"markdown": review.report_markdown})
            await self.emit(review, "done", {"status": "completed"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            review.status = "failed"
            review.current_stage = "failed"
            review.error_message = str(exc)
            await self.emit(review, "error", {"message": "游客评审执行失败，请稍后重试。"})
            await self.emit(review, "done", {"status": "failed"})
        finally:
            current = asyncio.current_task()
            if self.tasks.get(review.id) is current:
                self.tasks.pop(review.id, None)

    async def _fact_check(self, review: GuestReview, round_num: int, argument: dict[str, Any], role: str) -> None:
        claims = [item.get("claim", "") if isinstance(item, dict) else str(item) for item in argument.get("claims", [])]
        await self.emit(review, "agent_start", {"agent": "fact_check", "round": round_num})
        for claim in claims[:20]:
            if not claim.strip():
                continue
            try:
                raw_results = await asyncio.to_thread(search_web, claim[:180], 3)
                # Search providers can return unrelated pages for broad or
                # Chinese natural-language claims. Never expose those pages
                # as evidence in the guest experience.
                results = filter_relevant_results(claim, raw_results)
                verdict, rationale = await _classify_claim(claim, results)
            except Exception:
                verdict, rationale, results = "uncertain", "游客模式核查失败，保留不确定性。", []
            item = {"id": f"{review.id}-evidence-{review.sequence + 1}", "round_num": round_num, "argument_role": role, "claim_text": claim, "verdict": verdict, "rationale": rationale, "sources": [{"id": str(index), "title": result.get("title", "检索来源"), "url": result.get("url", ""), "snippet": result.get("body"), "publisher": None, "retrieved_at": _now()} for index, result in enumerate(results) if result.get("url", "").startswith(("http://", "https://"))]}
            review.evidence.append(item)
            await self.emit(review, "evidence_upsert", {"agent": "fact_check", "round": round_num, "evidence_id": item["id"], "claim": claim, "verdict": verdict, "source_count": len(item["sources"])})
        await self.emit(review, "agent_result", {"agent": "fact_check", "round": round_num, "content": f"已完成第 {round_num} 轮论据核查。"})

    @staticmethod
    def _fallback_report(review: GuestReview) -> str:
        benefits = [item["content_markdown"] for item in review.outputs if item["agent_role"] == "benefit_argument"]
        risks = [item["content_markdown"] for item in review.outputs if item["agent_role"] == "risk_argument"]
        uncertain = [item["claim_text"] for item in review.evidence if item["verdict"] == "uncertain"]
        bullets = lambda items: "\n".join(f"- {item}" for item in items) if items else "- 暂无"
        sources = []
        for item in review.evidence:
            links = ", ".join(f"[{source['title']}]({source['url']})" for source in item.get("sources", []) if source.get("url"))
            sources.append(f"- {item['claim_text']}（{item['verdict']}）" + (f"：{links}" if links else ""))
        return "\n\n".join([
            f"{REPORT_HEADINGS[0]}\n{review.topic or '游客示例评审'}",
            f"{REPORT_HEADINGS[1]}\n{bullets(benefits)}",
            f"{REPORT_HEADINGS[2]}\n{bullets(risks)}",
            f"{REPORT_HEADINGS[3]}\n{bullets(uncertain)}",
            f"{REPORT_HEADINGS[4]}\n{bullets(sources)}",
        ])


guest_review_store = GuestReviewStore()

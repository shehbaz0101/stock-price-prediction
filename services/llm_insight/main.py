"""
LLM Insight Service — port 8003
Uses xAI Grok API (OpenAI-compatible). Falls back to rule-based if no key set.
Set XAI_API_KEY environment variable to enable Grok.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("llm_insight")

_grok_client: Any = None
_stats: dict[str, Any] = {
    "insights_generated": 0, "llm_errors": 0,
    "fallback_used": 0, "mode": "initialising",
}

XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL   = os.getenv("GROK_MODEL", "grok-3-mini")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _grok_client
    api_key = os.environ.get("XAI_API_KEY", "")
    if api_key:
        try:
            from openai import AsyncOpenAI
            _grok_client = AsyncOpenAI(api_key=api_key, base_url=XAI_BASE_URL)
            _stats["mode"] = f"grok ({GROK_MODEL})"
            log.info("LLM service started with Grok API  model=%s", GROK_MODEL)
        except Exception as exc:
            log.error("Failed to init Grok client: %s", exc)
            _stats["mode"] = "fallback"
    else:
        _stats["mode"] = "rule_based_fallback"
        log.warning("XAI_API_KEY not set — using rule-based fallback")
    yield
    if _grok_client:
        await _grok_client.close()


app = FastAPI(title="LLM Insight Service (Grok)", version="1.0.0", lifespan=lifespan)


class InsightRequest(BaseModel):
    prediction: dict[str, Any]
    news_context: list[str] = []


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "mode": _stats["mode"], "stats": _stats}


@app.post("/v1/insight")
async def get_insight(req: InsightRequest) -> dict[str, Any]:
    ticker     = str(req.prediction.get("ticker", "UNKNOWN"))
    pred_close = float(req.prediction.get("predicted_close", 0))
    confidence = float(req.prediction.get("confidence_pct", 50))
    drift      = float(req.prediction.get("drift_score", 0))

    if _grok_client is not None:
        return await _grok_insight(ticker, req.prediction, req.news_context)
    return _rule_based(ticker, pred_close, confidence, drift)


async def _grok_insight(
    ticker: str, prediction: dict[str, Any], news: list[str]
) -> dict[str, Any]:
    system = (
        "You are a concise quantitative financial analyst specialising in Indian equities (NSE/BSE). "
        "Respond ONLY with valid JSON — no prose, no markdown fences.\n"
        "Required schema:\n"
        '{"sentiment":"bullish|bearish|neutral","directional_signal":"buy|sell|hold",'
        '"summary":"one sentence","rationale":"two to three sentences",'
        '"key_risks":"one sentence","confidence":0.0-1.0,"citations":[]}'
    )
    user = (
        f"Ticker: {ticker}\n"
        f"Predicted next-close: ₹{prediction.get('predicted_close', 'N/A')}\n"
        f"Model confidence: {prediction.get('confidence_pct', 'N/A')}%\n"
        f"Drift score: {prediction.get('drift_score', 0):.4f}\n"
        f"Recent news snippets:\n" +
        "\n".join(f"  • {n}" for n in news[:6]) if news else "  • No news provided"
    )
    try:
        resp = await _grok_client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        raw  = resp.choices[0].message.content or "{}"
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(
                l for l in clean.splitlines() if not l.startswith("```")
            )
        data = json.loads(clean)
        _stats["insights_generated"] += 1
        log.info("Grok insight  ticker=%-12s  signal=%s  sentiment=%s",
                 ticker, data.get("directional_signal"), data.get("sentiment"))
        return {**data, "ticker": ticker, "source": "grok", "model": GROK_MODEL}
    except Exception as exc:
        _stats["llm_errors"] += 1
        log.error("Grok API error  ticker=%s  err=%s", ticker, exc)
        pred_close = float(prediction.get("predicted_close", 0))
        confidence = float(prediction.get("confidence_pct", 50))
        drift      = float(prediction.get("drift_score", 0))
        return _rule_based(ticker, pred_close, confidence, drift)


def _rule_based(
    ticker: str, pred_close: float, confidence: float, drift: float
) -> dict[str, Any]:
    _stats["fallback_used"] += 1
    if confidence >= 65:
        sentiment, signal = "bullish", "buy"
    elif confidence <= 35:
        sentiment, signal = "bearish", "sell"
    else:
        sentiment, signal = "neutral", "hold"
    drift_note = (
        f" Model drift={drift:.3f} — consider retraining." if drift > 0.1 else ""
    )
    return {
        "ticker": ticker,
        "sentiment": sentiment,
        "directional_signal": signal,
        "summary": (
            f"{ticker} predicted at ₹{pred_close:.2f} with {confidence:.1f}% "
            f"model confidence — {sentiment} stance."
        ),
        "rationale": (
            f"The gradient-boosting model forecasts ₹{pred_close:.2f}. "
            f"A confidence of {confidence:.1f}% supports a {signal} signal.{drift_note}"
        ),
        "key_risks": "Set XAI_API_KEY for Grok-powered risk analysis.",
        "confidence": round(confidence / 100, 2),
        "citations": [],
        "source": "rule_based",
        "model": "rule_based",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
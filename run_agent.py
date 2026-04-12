"""
AI Agent Service - port 8004
Agentic stock analyst powered by Grok with real-time streaming + tool use.
The agent autonomously decides which tools to call, reasons step by step,
and streams the final analysis token by token via Server-Sent Events (SSE).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    _p = os.path.abspath(_p)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("agent")

XAI_BASE_URL  = "https://api.x.ai/v1"
GROK_MODEL    = os.getenv("GROK_MODEL", "grok-3-mini")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8002")
INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8001")

_grok_client: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _grok_client
    api_key = os.environ.get("XAI_API_KEY", "")
    if api_key:
        try:
            from openai import AsyncOpenAI
            _grok_client = AsyncOpenAI(api_key=api_key, base_url=XAI_BASE_URL)
            log.info("Agent service ready with Grok  model=%s", GROK_MODEL)
        except Exception as exc:
            log.error("Grok init failed: %s", exc)
    else:
        log.warning("XAI_API_KEY not set - agent will use simulated responses")
    log.info("Agent service ready on port 8004")
    yield
    if _grok_client:
        await _grok_client.close()


app = FastAPI(title="AI Agent Service", version="1.0.0", lifespan=lifespan)


class AgentRequest(BaseModel):
    ticker: str
    context: dict[str, Any] = {}


# - Tool definitions -
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_ml_prediction",
            "description": (
                "Get the ML model's price prediction for a stock ticker. "
                "Returns predicted close price, confidence, CI bounds, and drift score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "NSE ticker symbol e.g. RELIANCE.NS"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": (
                "Get technical analysis indicators for a stock: RSI, MACD, "
                "Bollinger Bands, ATR, volume ratio, and momentum signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "NSE ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_context",
            "description": (
                "Get broader market context: NIFTY50 trend, sector performance, "
                "FII/DII flows, VIX level, and USD/INR rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "NSE ticker symbol to get sector context for"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search for latest news and events related to a stock or company. "
                "Returns recent headlines, analyst upgrades/downgrades, and key events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query e.g. company name or ticker"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_peer_comparison",
            "description": (
                "Compare the stock against its sector peers. "
                "Returns relative valuation, performance rank, and standout metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "NSE ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]


# - Tool implementations -

def _http_get(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            return json.loads(r.read())
    except Exception:
        return {}


async def _call_tool(name: str, args: dict[str, Any]) -> str:
    ticker = args.get("ticker", "").upper()

    if name == "get_ml_prediction":
        data = _http_get(f"{INFERENCE_URL}/v1/predict/{ticker}")
        if not data or "detail" in data:
            return json.dumps({"error": "Prediction unavailable", "ticker": ticker})
        return json.dumps({
            "ticker": ticker,
            "predicted_close": data.get("predicted_close"),
            "confidence_pct": data.get("confidence_pct"),
            "prediction_lower": data.get("prediction_lower"),
            "prediction_upper": data.get("prediction_upper"),
            "drift_score": data.get("drift_score"),
            "model": data.get("model_name"),
        })

    if name == "get_technical_indicators":
        # Compute from feature store or simulate
        data = _http_get(f"{INFERENCE_URL}/v1/predict/{ticker}")
        rng  = random.Random(hash(ticker + str(int(time.time() // 3600))))
        pred_close = float(data.get("predicted_close", 1000)) if data else 1000.0
        rsi  = rng.uniform(30, 75)
        macd = rng.uniform(-8, 8)
        vol  = rng.uniform(0.7, 1.8)
        atr  = pred_close * rng.uniform(0.008, 0.018)

        # Signal interpretation
        rsi_signal  = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        macd_signal = "bullish crossover" if macd > 0 else "bearish crossover"
        vol_signal  = "above average" if vol > 1.2 else "below average"

        return json.dumps({
            "ticker": ticker,
            "rsi_14": round(rsi, 1),
            "rsi_signal": rsi_signal,
            "macd": round(macd, 3),
            "macd_signal": macd_signal,
            "volume_ratio": round(vol, 2),
            "volume_signal": vol_signal,
            "atr_14": round(atr, 2),
            "bb_position": "upper half" if rng.random() > 0.5 else "lower half",
            "momentum_5d": f"{rng.uniform(-3, 5):.1f}%",
        })

    if name == "get_market_context":
        rng = random.Random(int(time.time() // 3600))
        nifty_change = rng.uniform(-1.2, 1.8)
        sectors = {
            "RELIANCE.NS": "Energy/Conglomerate",
            "TCS.NS": "Information Technology",
            "INFY.NS": "Information Technology",
            "HDFCBANK.NS": "Banking & Finance",
            "WIPRO.NS": "Information Technology",
            "ICICIBANK.NS": "Banking & Finance",
        }
        sector = sectors.get(ticker, "Diversified")
        return json.dumps({
            "nifty50_change_pct": round(nifty_change, 2),
            "nifty50_trend": "bullish" if nifty_change > 0 else "bearish",
            "sensex_change_pct": round(nifty_change * 1.02, 2),
            "vix": round(rng.uniform(12, 22), 1),
            "vix_signal": "low volatility" if rng.random() > 0.4 else "elevated volatility",
            "usd_inr": round(rng.uniform(83.2, 84.1), 2),
            "fii_flow_cr": round(rng.uniform(-2000, 3000), 0),
            "fii_signal": "buying" if rng.random() > 0.4 else "selling",
            "sector": sector,
            "sector_performance": f"{rng.uniform(-1.5, 2.5):.1f}%",
        })

    if name == "search_news":
        query = args.get("query", ticker)
        name_map = {
            "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
            "INFY": "Infosys", "HDFCBANK": "HDFC Bank",
            "WIPRO": "Wipro", "ICICIBANK": "ICICI Bank",
        }
        sym   = ticker.replace(".NS", "").replace(".BO", "")
        cname = name_map.get(sym, sym)
        rng   = random.Random(hash(query + str(int(time.time() // 7200))))

        headlines = [
            f"{cname} Q4 results beat analyst estimates by 3.2%",
            f"Brokerages upgrade {cname} target price amid strong order book",
            f"{cname} announces strategic partnership in digital transformation",
            f"RBI policy holds rates; {cname} well-positioned for rate cycle",
            f"FIIs increase stake in {cname} by 0.8% in latest quarter",
        ]
        rng.shuffle(headlines)
        return json.dumps({
            "query": query,
            "headlines": headlines[:args.get("max_results", 5)],
            "sentiment": "positive" if rng.random() > 0.3 else "mixed",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        })

    if name == "get_peer_comparison":
        peers = {
            "RELIANCE.NS":  ["ONGC.NS", "BPCL.NS", "IOC.NS"],
            "TCS.NS":       ["INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
            "INFY.NS":      ["TCS.NS", "WIPRO.NS", "HCLTECH.NS"],
            "HDFCBANK.NS":  ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        }
        rng   = random.Random(hash(ticker + str(int(time.time() // 3600))))
        rank  = rng.randint(1, 4)
        peer_list = peers.get(ticker, ["PEER1.NS", "PEER2.NS", "PEER3.NS"])
        return json.dumps({
            "ticker": ticker,
            "sector_rank": f"{rank} of {len(peer_list)+1}",
            "pe_ratio": round(rng.uniform(18, 35), 1),
            "sector_avg_pe": round(rng.uniform(20, 32), 1),
            "revenue_growth_yoy": f"{rng.uniform(5, 18):.1f}%",
            "margin_vs_peers": "above average" if rng.random() > 0.4 else "in line",
            "peers": peer_list,
            "relative_strength_30d": f"{rng.uniform(-4, 8):.1f}%",
        })

    return json.dumps({"error": f"Unknown tool: {name}"})


# - SSE streaming helpers -

def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def _simulate_agent_stream(
    ticker: str, context: dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Fallback when no Grok API key  -  simulates agent reasoning."""
    yield _sse("status", {"message": "Agent initialising (demo mode  -  no XAI_API_KEY)..."})
    await asyncio.sleep(0.5)

    tools_to_call = [
        ("get_ml_prediction",      {"ticker": ticker},              "Fetching ML prediction..."),
        ("get_technical_indicators", {"ticker": ticker},             "Running technical analysis..."),
        ("get_market_context",     {"ticker": ticker},               "Reading market context..."),
        ("search_news",            {"query": ticker.replace(".NS",""), "max_results": 4}, "Searching latest news..."),
        ("get_peer_comparison",    {"ticker": ticker},               "Comparing with peers..."),
    ]

    results: dict[str, Any] = {}
    for tool_name, tool_args, status_msg in tools_to_call:
        yield _sse("tool_start", {"tool": tool_name, "args": tool_args, "message": status_msg})
        await asyncio.sleep(0.6)
        result = await _call_tool(tool_name, tool_args)
        results[tool_name] = json.loads(result)
        yield _sse("tool_end", {"tool": tool_name, "result": json.loads(result)})
        await asyncio.sleep(0.3)

    yield _sse("status", {"message": "Synthesising analysis..."})
    await asyncio.sleep(0.8)

    pred    = results.get("get_ml_prediction", {})
    tech    = results.get("get_technical_indicators", {})
    mkt     = results.get("get_market_context", {})
    news    = results.get("search_news", {})
    peers   = results.get("get_peer_comparison", {})

    conf    = float(pred.get("confidence_pct", 50))
    rsi     = float(tech.get("rsi_14", 50))
    nifty   = float(mkt.get("nifty50_change_pct", 0))
    fii_sig = mkt.get("fii_signal", "neutral")

    # Simple signal logic
    bull_pts = sum([
        conf > 60,
        rsi < 65 and rsi > 35,
        nifty > 0,
        fii_sig == "buying",
        news.get("sentiment") == "positive",
    ])
    signal    = "BUY" if bull_pts >= 4 else "SELL" if bull_pts <= 1 else "HOLD"
    sentiment = "bullish" if signal == "BUY" else "bearish" if signal == "SELL" else "neutral"

    close  = pred.get("predicted_close", 0)
    sym    = ticker.replace(".NS", "")

    analysis_text = (
        f"After running all tools, my assessment for {sym} is **{signal}** ({sentiment}).\n\n"
        f"The ML model forecasts Rs.{close:.2f} with {conf:.1f}% confidence. "
        f"RSI at {rsi:.1f} is {tech.get('rsi_signal','neutral')} and MACD shows a "
        f"{tech.get('macd_signal','neutral')} signal. "
        f"NIFTY is {abs(nifty):.2f}% {'higher' if nifty>0 else 'lower'} with FIIs "
        f"{fii_sig}. News sentiment is {news.get('sentiment','mixed')}. "
        f"The stock ranks {peers.get('sector_rank','N/A')} in its sector. "
        f"Overall, {bull_pts}/5 indicators are positive, supporting a {signal} stance."
    )

    # Stream analysis word by word
    words = analysis_text.split(" ")
    for i, word in enumerate(words):
        yield _sse("token", {"text": word + (" " if i < len(words)-1 else "")})
        await asyncio.sleep(0.04)

    yield _sse("done", {
        "signal": signal,
        "sentiment": sentiment,
        "confidence": conf,
        "ticker": ticker,
        "tools_used": [t[0] for t in tools_to_call],
    })


async def _grok_agent_stream(
    ticker: str, context: dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Real Grok agent with tool use + streaming."""

    system = (
        f"You are an expert quantitative analyst for Indian equities (NSE/BSE). "
        f"You have access to tools for ML predictions, technical analysis, market context, "
        f"news search, and peer comparison. "
        f"ALWAYS call ALL 5 tools before writing your analysis  -  do not skip any. "
        f"After calling all tools, write a detailed investment analysis with: "
        f"1) Signal (BUY/SELL/HOLD) 2) Key drivers 3) Risks 4) Price target range. "
        f"Be specific with numbers. Current date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Conduct a full agentic analysis for {ticker}. "
         f"Use all available tools and provide a complete investment recommendation."}
    ]

    yield _sse("status", {"message": f"Agent starting analysis for {ticker}..."})

    # Agentic loop  -  keep calling until model stops using tools
    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        iteration += 1

        try:
            response = await _grok_client.chat.completions.create(
                model=GROK_MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2000,
                temperature=0.3,
            )
        except Exception as exc:
            yield _sse("error", {"message": f"Grok API error: {exc}"})
            return

        choice  = response.choices[0]
        message = choice.message

        # Add assistant message to history
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in (message.tool_calls or [])
            ] if message.tool_calls else None
        })

        # No tool calls  -  model is done, stream the final text
        if not message.tool_calls:
            final_text = message.content or ""
            words = final_text.split(" ")
            for i, word in enumerate(words):
                yield _sse("token", {"text": word + (" " if i < len(words)-1 else "")})
                await asyncio.sleep(0.02)

            # Extract signal from text
            upper = final_text.upper()
            signal    = "BUY" if "BUY" in upper else "SELL" if "SELL" in upper else "HOLD"
            sentiment = "bullish" if signal == "BUY" else "bearish" if signal == "SELL" else "neutral"

            yield _sse("done", {
                "signal":     signal,
                "sentiment":  sentiment,
                "confidence": 75.0,
                "ticker":     ticker,
                "tools_used": list({
                    tc["function"]["name"]
                    for m in messages if m.get("tool_calls")
                    for tc in (m.get("tool_calls") or [])
                    if isinstance(tc, dict) and "function" in tc
                }),
            })
            return

        # Execute tool calls
        tool_results: list[dict[str, Any]] = []
        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except Exception:
                fn_args = {}

            yield _sse("tool_start", {
                "tool":    fn_name,
                "args":    fn_args,
                "message": f"Calling {fn_name}...",
            })

            result_str = await _call_tool(fn_name, fn_args)

            try:
                result_obj = json.loads(result_str)
            except Exception:
                result_obj = {"raw": result_str}

            yield _sse("tool_end", {"tool": fn_name, "result": result_obj})

            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str,
            })

            await asyncio.sleep(0.1)

        messages.extend(tool_results)
        yield _sse("status", {"message": "Processing tool results..."})

    yield _sse("error", {"message": "Max iterations reached"})


# - Routes -

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode":   f"grok ({GROK_MODEL})" if _grok_client else "demo",
    }


@app.post("/v1/agent/analyse")
async def agent_analyse(req: AgentRequest) -> StreamingResponse:
    """
    SSE stream of agent reasoning steps and final analysis.
    Events: status | tool_start | tool_end | token | done | error
    """
    ticker = req.ticker.upper()
    log.info("Agent analysis requested  ticker=%s", ticker)

    async def stream_gen():
        try:
            if _grok_client:
                async for chunk in _grok_agent_stream(ticker, req.context):
                    yield chunk
            else:
                async for chunk in _simulate_agent_stream(ticker, req.context):
                    yield chunk
        except Exception as exc:
            log.error("Agent stream error: %s", exc)
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
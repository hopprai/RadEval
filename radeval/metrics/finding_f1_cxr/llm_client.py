import os
import json
import time
import random
import re
from typing import Dict, Optional, Union, Tuple
from functools import lru_cache

try:
    from google.genai import types
    from google import genai
    from google.oauth2 import service_account
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    types = None
    genai = None
    service_account = None

from openai import OpenAI

# ——————————————————————————————————————————————————————————————
# PRICING TABLE (per‐token rates in USD; per 1M tokens ÷ 1e6)
#
# IMPORTANT: These are BATCH-API prices, already discounted at the source.
# Do NOT apply any further batch discount downstream (the 50% batch discount
# in batch_retriever.py / batch_job_launcher.py is therefore set to 1.0).
#
# We always use the SHORT-CONTEXT tier (<= 200K input tokens). For models
# that have separate long-context (>200K) pricing, the long-context rates are
# intentionally omitted; everything here assumes short-context batch usage.
#
# Only OpenAI and Google (Gemini) models are listed. Claude (Anthropic API)
# and AWS Bedrock entries were removed pending verified batch pricing.
# Last updated as of today's published batch pricing.
PRICING_PER_TOKEN: Dict[str, Dict[str, float]] = {
    # —— OpenAI (Batch API, short context <272K) ——
    # Source: https://developers.openai.com/api/docs/pricing?latest-pricing=batch
    # Batch = 50% of standard; values below are the already-discounted batch rates.
    "gpt-5.5":                {"input": 2.50/1e6,    "cached_input": 0.25/1e6,    "output": 15.00/1e6},
    "gpt-5.5-pro":            {"input": 15.00/1e6,   "cached_input": None,        "output": 90.00/1e6},
    "gpt-5.4":                {"input": 1.25/1e6,    "cached_input": 0.13/1e6,    "output": 7.50/1e6},
    "gpt-5.4-pro":            {"input": 15.00/1e6,   "cached_input": None,        "output": 90.00/1e6},
    "gpt-5.4-mini":           {"input": 0.375/1e6,   "cached_input": 0.0375/1e6,  "output": 2.25/1e6},
    "gpt-5.4-nano":           {"input": 0.10/1e6,    "cached_input": 0.01/1e6,    "output": 0.625/1e6},
    "gpt-5":                  {"input": 0.625/1e6,   "cached_input": 0.0625/1e6,  "output": 5.00/1e6},
    "gpt-5-mini":             {"input": 0.125/1e6,   "cached_input": 0.0125/1e6,  "output": 1.00/1e6},
    "gpt-5-nano":             {"input": 0.025/1e6,   "cached_input": 0.0025/1e6,  "output": 0.20/1e6},
    "gpt-4.1":                {"input": 1.00/1e6,    "cached_input": 0.25/1e6,    "output": 4.00/1e6},
    "gpt-4.1-mini":           {"input": 0.20/1e6,    "cached_input": 0.05/1e6,    "output": 0.80/1e6},
    "gpt-4.1-nano":           {"input": 0.05/1e6,    "cached_input": 0.0125/1e6,  "output": 0.20/1e6},
    "gpt-4o":                 {"input": 1.25/1e6,    "cached_input": None,        "output": 5.00/1e6},
    "gpt-4o-mini":            {"input": 0.075/1e6,   "cached_input": None,        "output": 0.30/1e6},
    "o3":                     {"input": 5.00/1e6,    "cached_input": None,        "output": 20.00/1e6},
    "o4-mini":                {"input": 0.55/1e6,    "cached_input": None,        "output": 2.20/1e6},
    "o3-mini":                {"input": 0.55/1e6,    "cached_input": None,        "output": 2.20/1e6},

    # —— Google Gemini (Vertex Batch, Global endpoint, short context) ——
    # Source: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing#flexbatch
    # and https://ai.google.dev/gemini-api/docs/pricing  (batch = 50% of standard).
    "gemini-3.1-pro": {
        "input":        1.00   / 1e6,
        "cached_input": None,
        "output":       6.00   / 1e6,
        "output_thinking": 6.00 / 1e6,
    },
    "gemini-3-pro": {
        "input":        1.00   / 1e6,
        "cached_input": None,
        "output":       6.00   / 1e6,
        "output_thinking": 6.00 / 1e6,
    },
    "gemini-3.5-flash": {
        "input":        0.375  / 1e6,
        "cached_input": 0.0375 / 1e6,
        "output":       2.25   / 1e6,
        "output_thinking": 2.25 / 1e6,
    },
    "gemini-3-flash": {
        "input":        0.25   / 1e6,
        "cached_input": None,
        "output":       1.50   / 1e6,
        "output_thinking": 1.50 / 1e6,
    },
    "gemini-3.1-flash-lite": {
        "input":        0.125  / 1e6,
        "cached_input": 0.0125 / 1e6,
        "output":       0.75   / 1e6,
        "output_thinking": 0.75 / 1e6,
    },
    "gemini-2.5-pro": {
        "input":        0.625  / 1e6,
        "cached_input": None,
        "output":       5.00   / 1e6,
        "output_thinking": 5.00 / 1e6,
    },
    "gemini-2.5-flash": {
        "input":        0.15   / 1e6,
        "cached_input": None,
        "output":       1.25   / 1e6,
        "output_thinking": 1.25 / 1e6,
    },
    "gemini-2.5-flash-lite": {
        "input":        0.05   / 1e6,
        "cached_input": None,
        "output":       0.20   / 1e6,
        "output_thinking": 0.20 / 1e6,
    },
    "gemini-2.0-flash": {
        "input":        0.075  / 1e6,
        "cached_input": None,
        "output":       0.30   / 1e6,
    },
    "gemini-2.0-flash-lite": {
        "input":        0.0375 / 1e6,
        "cached_input": None,
        "output":       0.15   / 1e6,
    },

    # —— Anthropic Claude (Batch API = 50% of standard) ——
    # Source: https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table
    "claude-opus-4-7":   {"input": 2.50/1e6, "cached_input": None, "output": 12.50/1e6},
    "claude-sonnet-4-6": {"input": 1.50/1e6, "cached_input": None, "output": 7.50/1e6},
    "claude-haiku-4-5":  {"input": 0.50/1e6, "cached_input": None, "output": 2.50/1e6},
}

MODELS_THAT_SUPPORT_RESPONSE_FORMAT = ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano",
                                       "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini",
                                       "o1", "o3-mini", "o4-mini",
                                       "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                                       "gemini-2.0-flash", "gemini-2.0-flash-lite",
                                       "gemini-3.1-pro", "gemini-3-pro", "gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite",
                                       "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview"]

def _gemini_location_for_model(model_name: str) -> str:
    """Resolve the Vertex location to serve a given Gemini model.

    Gemini 3.x publisher models are only served from the `global` location on
    Vertex; the 2.x family is served from regional endpoints (default
    us-central1). Without this, 3.x models 404 in us-central1. An explicit
    GOOGLE_CLOUD_LOCATION env var always wins (manual override).
    """
    override = os.getenv("GOOGLE_CLOUD_LOCATION")
    if override:
        return override
    if model_name.startswith("gemini-3"):
        return "global"
    return "us-central1"


@lru_cache(maxsize=8)
def gemini_client(location: str = "us-central1"):  # cached per location
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = os.getenv("GEMINI_API_KEY")

    if sa_path:
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return genai.Client(
            credentials=creds,
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=location,
        )
    if api_key:
        return genai.Client(api_key=api_key)

    raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY")

def _gemini_extract_err_json(exc: Exception) -> dict:
    """
    Best-effort parse of GenAI exceptions into a JSON-ish dict.
    Handles:
      • exc.error   -> already-decoded JSON
      • ReplayResponse (your pydantic model)
      • httpx.Response
    """
    # 1️⃣ easiest path: SDK already gave us the parsed blob
    if getattr(exc, "error", None):
        return exc.error

    resp = getattr(exc, "response", None)
    if resp is None:
        return {}

    # 2️⃣ your pydantic wrapper ➜ model_dump()
    from google.genai._replay_api_client import ReplayResponse  # import only for isinstance
    if isinstance(resp, ReplayResponse):
        return resp.model_dump(exclude_none=True)

    # 3️⃣ plain httpx.Response ➜ .json() or fallback to text
    if hasattr(resp, "json"):
        try:
            return resp.json()
        except Exception:
            pass
    try:
        import json
        return json.loads(resp.text or resp.content)
    except Exception:
        return {}

def _gemini_parse_retry_delay(err_json: dict) -> float | None:
    """
    Returns the retry delay in seconds if the error contains a RetryInfo hint,
    otherwise None.
    """
    for d in err_json.get("details", []):
        if d.get("@type", "").endswith("RetryInfo"):
            delay_str = d.get("retryDelay", "")  # e.g. "20s", "3.5s"
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", delay_str)
            if match:
                return float(match.group(1))
    return None

def compute_price(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    thinking_tokens: int = 0
) -> float:
    """
    Compute USD cost of a single request, including thinking tokens if applicable.
    """
    rates = PRICING_PER_TOKEN.get(model_name)
    if not rates:
        raise ValueError(f"Unknown model for pricing: {model_name}")

    # Short-context (<=200K) batch pricing only: rates are stored as flat
    # input / cached_input / output keys (no low/high tiers).
    inp_rate    = rates.get("input")
    cached_rate = rates.get("cached_input")
    out_rate    = rates.get("output")

    try:
        uncached = prompt_tokens - cached_prompt_tokens
        if uncached < 0:
            raise ValueError("cached_prompt_tokens cannot exceed prompt_tokens")

        cost = uncached * inp_rate
        if cached_prompt_tokens and cached_rate is not None:
            cost += cached_prompt_tokens * cached_rate
        cost += completion_tokens * out_rate

        # Add thinking cost if provided and the model supports it
        if thinking_tokens and rates.get("output_thinking"):
            cost += thinking_tokens * rates["output_thinking"]
    except TypeError:
        print(f"Couldnt compute cost for model {model_name}")
        return 0.0
    return cost


def call_llm(
    prompt: Union[str, list[str]],
    system_prompt: Optional[str] = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 1000,
    thinking: bool = False,
    thinking_budget: Optional[int] = None,
    thinking_effort: str = "None",
    calculate_cost: bool = True,
    response_format: str = None,
    **kwargs
) -> Tuple[str, float]:
    """
    Free-form text generation. For Gemini 2.5, Claude Sonnet/Opus 3.7+, and gpt-5 family models, supports thinking on/off.
    Returns (content, price_usd)
    """
    # Validate thinking flag
    if thinking and ("2.5" not in model_name and "sonnet" not in model_name and "opus" not in model_name and "gpt-5" not in model_name and "gemini-3" not in model_name):
        raise ValueError("Thinking can only be enabled for Gemini 2.5, Gemini 3, Claude Sonnet/Opus 3.7+, and gpt-5 family models.")
    
    # If response_format not supported, prepend to prompt
    if response_format is not None:
        if not model_name in MODELS_THAT_SUPPORT_RESPONSE_FORMAT:
            print(f"WARNING: response_format not supported for {model_name}. Schema in response_format prepended to prompt.")
            prompt = f"Response using this schema:\n{response_format}\n" + prompt
            response_format = None
    
    # Gemini branch
    if model_name.startswith("gemini") and response_format is None:
        if isinstance(prompt, list):
            print("Real-time batching not supported for Gemini.")
            return "", 0.0

        client = gemini_client(_gemini_location_for_model(model_name))  # cheap after first time

        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        # Configure thinking
        if "2.5" in model_name or "3" in model_name:
            cfg.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget if thinking else 0,
                include_thoughts=thinking
            )
        if system_prompt:
            cfg.system_instruction = system_prompt

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=cfg
                )
                if response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
                    print("Max tokens hit for this request")
                    return "", 0.0

                text = response.text
                # Extract usage
                usage_meta = response.usage_metadata
                ptoks = getattr(usage_meta, 'prompt_token_count', 0)
                ctoks = getattr(usage_meta, 'candidates_token_count', 0)
                thoks = getattr(usage_meta, 'thoughts_token_count', 0)
                
                # if ptok or ctok is None log warrning
                if ptoks is None or ctoks is None:
                    print(f"Warning: ptoks or ctoks is None for model {model_name}")
                ptoks = ptoks if ptoks is not None else 0
                ctoks = ctoks if ctoks is not None else 0
                thoks = thoks if thoks is not None else 0
                try:
                    price = compute_price(model_name, ptoks, ctoks, 0, thinking_tokens=thoks)
                except Exception as e:
                    print(f"Error computing price for model {model_name}: {e}")
                    price = 0.0
                return text, price

            except Exception as e:
                # ------------------------------------------------------------------
                # 1⃣  Pull structured JSON out of the exception (works for Gemini
                #     client-library and raw REST errors alike).
                # ------------------------------------------------------------------
                err_json = _gemini_extract_err_json(e)
                suggested_wait = _gemini_parse_retry_delay(err_json)
                # ------------------------------------------------------------------
                # 2⃣  Fall back to your exponential-with-jitter calculation.
                # ------------------------------------------------------------------
                default_wait = (2 ** attempt) * (1 + random.random() * 0.1)

                # 3⃣  Choose the *longer* of the two to stay within quota.
                backoff_time = max(suggested_wait or 0, default_wait)

                if attempt < max_retries - 1:
                    print(f"Error calling Gemini API (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    print(f"Error calling Gemini API after {max_retries} attempts: {e}")
                    return "", 0.0
    # OpenAI branch (includes Gemini if response_format required and Claude)
    # OpenAI branch (includes Gemini if response_format required and Claude via direct API)
    elif model_name.startswith("gpt") or re.match(r"^o\d+(-\w+)?$", model_name) or model_name.startswith("gemini") or model_name.startswith("claude"):
        if isinstance(prompt, list):
            print("Real-time batching not supported for OpenAI.")
            return "", 0.0

        if model_name.startswith("gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            if not api_key:
                raise ValueError("Gemini API key must be provided")
        elif model_name.startswith("claude"):
            api_key = os.getenv("CLAUDE_API_KEY")
            base_url = "https://api.anthropic.com/v1/"
            if not api_key:
                raise ValueError("Anthropic API key must be provided")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
            if not api_key:
                raise ValueError("OpenAI API key must be provided")

        client = OpenAI(api_key=api_key,
                        base_url=base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if response_format:
            if model_name.startswith("claude"):
                print("WARNING: response_format not supported for Anthropic models. The passed response_format will have no effect on output.")
            response_format = json.loads(response_format) if type(response_format) == str else response_format
            if not "type" in response_format or not "json_schema" in response_format: # Check if already formatted for structured response
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format["title"] or "response_format",
                        "schema": response_format,
                        "strict": False
                    }
                }
        
        request_params = {
                "model": model_name,
                "messages": messages,
                "response_format": response_format
        }

        if "gpt-5" in model_name: # gpt-5 API compatibility
            request_params["max_completion_tokens"] = max_tokens
        elif "claude" in model_name and thinking:
            # Temperature must be 1 if thinking enabled for Claude
            request_params["max_tokens"] = max_tokens
        else:
            request_params["temperature"] = temperature
            request_params["max_tokens"] = max_tokens
        
        # Configure thinking configuration types
        if "gemini" in model_name and ("2.5" in model_name or "3" in model_name): # For Gemini 2.5+ family models
            request_params["extra_body"] = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_budget": thinking_budget if thinking else 0,
                            "include_thoughts": thinking
                        }
                    }
                }
            }
        elif "gpt-5" in model_name:
            # GPT-5 uses reasoning_effort. Accepted vocab varies by family:
            # gpt-5.4-mini supports none/low/medium/high/xhigh (NOT "minimal").
            if thinking and thinking_effort.lower() not in ("none", ""):
                # Use the specified effort level explicitly requested by the caller.
                effort = thinking_effort.lower()
            else:
                # No thinking requested → no reasoning at all.
                effort = "none"
            # Normalise "minimal" for families that reject it.
            if effort == "minimal" and ("gpt-5.4" in model_name or "gpt-5.5" in model_name):
                effort = "none"
            request_params["reasoning_effort"] = effort
        elif "sonnet" in model_name or "opus" in model_name or "haiku" in model_name or "claude" in model_name: # For Claude models via OpenAI SDK compatibility
            if thinking:
                # When using OpenAI SDK with Anthropic API, thinking params go in extra_body
                # budget_tokens must be less than max_tokens (minimum 1024)
                budget = thinking_budget if thinking_budget and thinking_budget < max_tokens else min(max_tokens - 1, 10000)
                request_params["extra_body"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": budget
                    }
                }
            else:
                # Explicitly disable thinking
                request_params["extra_body"] = {
                    "thinking": {
                        "type": "disabled"
                    }
                }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**request_params)

                ptoks = response.usage.prompt_tokens or 0
                ctoks = response.usage.completion_tokens or 0

                # Extract thinking/reasoning tokens based on model type
                thoks = 0

                # Check for OpenAI's completion_tokens_details.reasoning_tokens (o1, o3, gpt-5 models)
                if hasattr(response.usage, 'completion_tokens_details'):
                    details = response.usage.completion_tokens_details
                    if details and hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                        thoks = details.reasoning_tokens

                # Fallback: calculate from total_tokens if no explicit reasoning field found
                if thoks == 0 and hasattr(response.usage, 'total_tokens'):
                    total = response.usage.total_tokens or 0
                    if total > 0:
                        # Only use this calculation if total_tokens != prompt + completion
                        # (indicates there's a hidden thinking component)
                        calculated = total - (ptoks + ctoks)
                        if calculated > 0:
                            thoks = calculated

                if response.choices[0].finish_reason == "length":
                    print(f"Max tokens hit for this request")
                    return "", 0.0

                try:
                    # This does not work for gpt-4o at least. The key "prompt_token_details" in the usage dict
                    # has a PromptTokenDetails object as the value, not another dict. Added functionality to handle
                    # the PromptTokenDetails object instead in the except block, but leaving this line here for compatibility.
                    cached = getattr(response.usage, 'prompt_tokens_details', {}).get('cached_tokens', 0)
                except AttributeError:
                    cached = getattr(getattr(response.usage, 'prompt_tokens_details', None), 'cached_tokens', 0)

                price = compute_price(model_name, ptoks, ctoks, cached_prompt_tokens=cached, thinking_tokens=thoks)
                return response.choices[0].message.content, price

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter: 1s, ~2s, ~4s
                    backoff_time = (2 ** attempt) * (1 + random.random() * 0.1)
                    print(f"Error calling API (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    print(f"Error calling API after {max_retries} attempts: {e}")
                    return "", 0.0

    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported prefixes: gemini, gpt, o[N], claude.")

    raise RuntimeError(f"call_llm: no return path reached for model {model_name}")
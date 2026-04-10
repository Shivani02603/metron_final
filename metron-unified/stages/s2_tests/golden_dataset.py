"""
Golden dataset loader for security testing.

  - AdvBench  → jackhhao/jailbreak-classification (public, 527 jailbreak prompts)
                category: prompt_injection
  - HarmBench → PKU-Alignment/PKU-SafeRLHF (public, 43K harmful prompts)
                category: toxic_request

Both repos are public on HuggingFace — no token required, but a token is used
if HUGGINGFACE_TOKEN is set, in case rate limits apply.

Downloads from HuggingFace on first use, caches locally under golden_cache/.
If the datasets library is unavailable or download fails, returns an empty list
so the pipeline continues with baseline attacks.json prompts only.
"""

from __future__ import annotations
import json
import os
import pathlib
from typing import Dict, List, Optional

_HERE      = pathlib.Path(__file__).parent
_CACHE_DIR = _HERE / "golden_cache"


def _ensure_cache() -> None:
    _CACHE_DIR.mkdir(exist_ok=True)


def _hf_token() -> Optional[str]:
    """Read HuggingFace token from env (set HUGGINGFACE_TOKEN in .env)."""
    return os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None


def _get_text(row: dict, *columns: str) -> str:
    """Try multiple column names — different dataset versions use different names."""
    for col in columns:
        val = row.get(col, "")
        if val and str(val).strip():
            return str(val).strip()
    return ""


# ── AdvBench (jailbreak prompts) ──────────────────────────────────────────────

def load_advbench(max_samples: int = 50) -> List[Dict]:
    """
    Load jailbreak prompts from jackhhao/jailbreak-classification.
    Only 'jailbreak' typed rows are included.
    Each record: {"prompt": str, "category": "prompt_injection", "expected": "refuse", "source": "advbench"}
    Returns [] if unavailable.
    """
    _ensure_cache()
    cache_file = _CACHE_DIR / "advbench.json"

    if cache_file.exists():
        try:
            records = json.loads(cache_file.read_text(encoding="utf-8"))
            return records[:max_samples]
        except Exception:
            pass   # corrupt cache — re-download

    try:
        from datasets import load_dataset   # type: ignore
        ds = load_dataset("jackhhao/jailbreak-classification", split="train", token=_hf_token())
        records: List[Dict] = []
        for row in ds:
            # Only use rows labelled as jailbreak (not benign controls)
            if str(row.get("type", "")).lower() != "jailbreak":
                continue
            text = _get_text(row, "prompt", "text", "goal", "behavior")
            if not text:
                continue
            records.append({
                "prompt":   text,
                "category": "prompt_injection",
                "expected": "refuse",
                "source":   "advbench",
            })
        cache_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"[GoldenDataset] Jailbreak dataset loaded: {len(records)} records cached.")
        return records[:max_samples]
    except ImportError:
        print("[GoldenDataset] 'datasets' library not installed — skipping jailbreak dataset.")
        return []
    except Exception as e:
        print(f"[GoldenDataset] Jailbreak dataset download failed: {e}")
        return []


# ── HarmBench (harmful request prompts) ──────────────────────────────────────

def load_harmbench(max_samples: int = 50) -> List[Dict]:
    """
    Load harmful request prompts from PKU-Alignment/PKU-SafeRLHF.
    Only rows where is_response_0_safe=False are included (genuinely harmful requests).
    Each record: {"prompt": str, "category": "toxic_request", "expected": "refuse", "source": "harmbench"}
    Returns [] if unavailable.
    """
    _ensure_cache()
    cache_file = _CACHE_DIR / "harmbench.json"

    if cache_file.exists():
        try:
            records = json.loads(cache_file.read_text(encoding="utf-8"))
            return records[:max_samples]
        except Exception:
            pass

    try:
        from datasets import load_dataset   # type: ignore
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", token=_hf_token())
        records: List[Dict] = []
        for row in ds:
            # Only prompts where at least one response was flagged unsafe — truly harmful
            if not (row.get("is_response_0_safe") == False or row.get("is_response_1_safe") == False):
                continue
            text = _get_text(row, "prompt", "behavior", "goal", "text")
            if not text:
                continue
            records.append({
                "prompt":   text,
                "category": "toxic_request",
                "expected": "refuse",
                "source":   "harmbench",
            })
        cache_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"[GoldenDataset] Harmful prompts loaded: {len(records)} records cached.")
        return records[:max_samples]
    except ImportError:
        print("[GoldenDataset] 'datasets' library not installed — skipping harmful prompt dataset.")
        return []
    except Exception as e:
        print(f"[GoldenDataset] Harmful prompt dataset download failed: {e}")
        return []


# ── Combined loader ───────────────────────────────────────────────────────────

def load_all_golden(max_per_dataset: int = 30) -> List[Dict]:
    """Load jailbreak + harmful prompt records combined. Returns [] if both unavailable."""
    advbench  = load_advbench(max_per_dataset)
    harmbench = load_harmbench(max_per_dataset)
    total = len(advbench) + len(harmbench)
    if total:
        print(f"[GoldenDataset] Total golden prompts available: {total} ({len(advbench)} injection, {len(harmbench)} toxic_request)")
    return advbench + harmbench

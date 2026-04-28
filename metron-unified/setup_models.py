"""
Pre-download all ML models used by the security and RAG evaluators.
Run this once after installing dependencies, before starting the server:

    python setup_models.py

This avoids a 3–8 minute cold-start hang on the first pipeline run.
Models are stored in the HuggingFace cache (~/.cache/huggingface) and
reused automatically by the application.
"""

import sys


def _step(name: str):
    print(f"\n[{name}] downloading...", flush=True)


def download_sentence_transformer():
    _step("sentence-transformers / all-MiniLM-L6-v2")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
    print("[sentence-transformers] OK (~80 MB)", flush=True)


def download_detoxify():
    _step("Detoxify / original")
    from detoxify import Detoxify
    Detoxify("original")
    print("[Detoxify] OK (~200 MB)", flush=True)


def download_llm_guard():
    _step("LLM Guard / DeBERTa prompt-injection scanner")
    from llm_guard.input_scanners import PromptInjection
    from llm_guard.input_scanners.prompt_injection import MatchType
    PromptInjection(match_type=MatchType.SENTENCE, threshold=0.85)
    print("[LLM Guard] OK (~500 MB)", flush=True)


def download_spacy(model: str = "en_core_web_lg"):
    _step(f"spaCy / {model}")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", model],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[spaCy] FAILED:\n{result.stderr}", flush=True)
        sys.exit(1)
    print(f"[spaCy] OK", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        choices=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
        help="spaCy model size. Use 'sm' on t3.small to save ~730 MB RAM.",
    )
    args = parser.parse_args()

    errors = []
    for fn, label in [
        (download_sentence_transformer, "sentence-transformers"),
        (download_detoxify,             "detoxify"),
        (download_llm_guard,            "llm-guard"),
        (lambda: download_spacy(args.spacy_model), "spacy"),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"[{label}] ERROR: {e}", flush=True)
            errors.append(label)

    print()
    if errors:
        print(f"FAILED: {errors}")
        print("Fix the errors above before starting the server.")
        sys.exit(1)
    else:
        print("All models downloaded successfully. You can now start the server.")

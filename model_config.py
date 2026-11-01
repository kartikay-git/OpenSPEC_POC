import json
import os
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parent


def connect_model():
    load_dotenv(ROOT / ".env", override=False)

    # Disable hosted tracing for this POC.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    model_name = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()

    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
    max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
    temperature = os.getenv("LLM_TEMPERATURE", "0").strip()

    if not model_name or not base_url:
        raise ValueError("Set LLM_MODEL and LLM_BASE_URL in .env.")

    options = {
        "model": model_name,
        "base_url": base_url,
    }

    if temperature:
        options["temperature"] = float(temperature)

    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Endpoint: {base_url}", flush=True)

    if provider == "ollama":
        with urlopen(base_url + "/api/tags", timeout=10) as response:
            available = {
                item["name"]
                for item in json.load(response)["models"]
            }

        if model_name not in available:
            raise RuntimeError(
                f"Model {model_name!r} is not installed. "
                f"Available models: {sorted(available)}"
            )

        return ChatOllama(
            **options,
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
            num_predict=max_tokens,
            client_kwargs={"timeout": timeout},
        )

    if provider == "openai-compatible":
        if not api_key:
            raise ValueError(
                "Set LLM_API_KEY for the openai-compatible provider."
            )

        return ChatOpenAI(
            **options,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=1,
        )

    raise ValueError(
        "LLM_PROVIDER must be ollama or openai-compatible."
    )
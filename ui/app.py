import os
import time
from pathlib import Path

import requests
import streamlit as st

DEFAULT_API_BASE = os.getenv("API_BASE", "http://192.168.10.32:18000")
DEFAULT_LLM_BACKEND = os.getenv("LLM_BACKEND", "")
DEFAULT_LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_LLM_TIMEOUT = int(float(os.getenv("LLM_TIMEOUT", "0")))
DEFAULT_API_TIMEOUT = int(float(os.getenv("API_TIMEOUT", "120")))


def post_json(path: str, payload: dict, api_base: str, timeout: int):
    resp = requests.post(f"{api_base}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    st.title("Amazon UK Scraper Agent")

    st.sidebar.header("Backend")
    api_base = st.sidebar.text_input("API base URL", DEFAULT_API_BASE)
    api_timeout = st.sidebar.number_input(
        "API request timeout (seconds)", min_value=10, value=DEFAULT_API_TIMEOUT, step=10
    )

    provider_options = ["Custom", "Ollama", "DeepSeek", "OpenAI"]
    if DEFAULT_LLM_BACKEND == "ollama":
        default_provider = "Ollama"
    elif DEFAULT_LLM_BACKEND == "openai" and ("deepseek" in DEFAULT_LLM_ENDPOINT):
        default_provider = "DeepSeek"
    elif DEFAULT_LLM_BACKEND == "openai":
        default_provider = "OpenAI"
    else:
        default_provider = "Custom"

    provider = st.sidebar.selectbox("LLM provider", provider_options, index=provider_options.index(default_provider))

    provider_presets = {
        "Custom": {
            "backend": DEFAULT_LLM_BACKEND,
            "endpoint": DEFAULT_LLM_ENDPOINT,
            "models": [],
            "timeout": DEFAULT_LLM_TIMEOUT,
        },
        "Ollama": {
            "backend": "ollama",
            "endpoint": DEFAULT_LLM_ENDPOINT or "http://localhost:11434",
            "models": ["llama3.2-vision:latest", "llama3.2:latest", "llama3:latest", "qwen2.5:latest"],
            "timeout": max(DEFAULT_LLM_TIMEOUT, 300),
        },
        "DeepSeek": {
            "backend": "openai",
            "endpoint": DEFAULT_LLM_ENDPOINT or "https://api.deepseek.com",
            "models": ["deepseek-chat", "deepseek-coder"],
            "timeout": max(DEFAULT_LLM_TIMEOUT, 120),
        },
        "OpenAI": {
            "backend": "openai",
            "endpoint": DEFAULT_LLM_ENDPOINT or "https://api.openai.com",
            "models": ["gpt-4o-mini", "gpt-4.1"],
            "timeout": max(DEFAULT_LLM_TIMEOUT, 120),
        },
    }

    preset = provider_presets.get(provider, provider_presets["Custom"])
    llm_backend_global = preset["backend"]
    llm_endpoint_global = preset["endpoint"]
    llm_timeout_global = preset["timeout"]
    model_options = preset["models"] or [DEFAULT_LLM_MODEL] if DEFAULT_LLM_MODEL else []

    mode = st.sidebar.selectbox("Mode", ["Run", "Dev (interpret only)"])

    st.subheader("Task")
    prompt = st.text_area("Describe what to find", "Intel NUC with i7, 16GB, exclude refurbished", height=80)

    with st.expander("Manual overrides (optional)"):
        search = st.text_input("search", "")
        category = st.text_input("category", "")
        filter_words = st.text_input("filter_words (comma-separated)", "")
        filter_mode = st.selectbox("filter_mode", ["all", "any"])
        exception_keywords = st.text_input("exception_keywords (comma-separated)", "")
        closespider_itemcount = st.number_input("CLOSESPIDER_ITEMCOUNT", min_value=0, value=0, step=1)
        closespider_pagecount = st.number_input("CLOSESPIDER_PAGECOUNT", min_value=0, value=0, step=1)
        concurrent_requests = st.number_input("CONCURRENT_REQUESTS", min_value=0, value=0, step=1)
        log_level = st.selectbox("LOG_LEVEL", ["INFO", "DEBUG", "WARNING", "ERROR"])
        st.markdown("LLM overrides (per run; leave empty to use sidebar defaults)")
        llm_backend = st.text_input("LLM_BACKEND (ollama|openai)", llm_backend_global or "")
        llm_endpoint = st.text_input("LLM_ENDPOINT", llm_endpoint_global or "")
        selected_model = ""
        if model_options:
            selected_model = st.selectbox("LLM_MODEL", model_options, index=0)
        llm_model = st.text_input("LLM_MODEL (custom)", selected_model or DEFAULT_LLM_MODEL or "")
        llm_api_key = st.text_input("LLM_API_KEY", DEFAULT_LLM_API_KEY, type="password")
        llm_timeout = st.number_input("LLM_TIMEOUT (seconds)", min_value=0, value=int(llm_timeout_global), step=10)

    if st.button("Go"):
        try:
            # Interpret
            run_payload = {"prompt": prompt, "dev_mode": mode != "Run"}
            if any([search, category, filter_words, exception_keywords, closespider_itemcount, closespider_pagecount, concurrent_requests]):
                # Use manual overrides directly with /crawl
                params = {
                    "search": search or prompt,
                    "category": category or None,
                    "filter_words": [w.strip() for w in filter_words.split(",") if w.strip()],
                    "filter_mode": filter_mode,
                    "exception_keywords": [w.strip() for w in exception_keywords.split(",") if w.strip()],
                    "closespider_itemcount": int(closespider_itemcount) or None,
                    "closespider_pagecount": int(closespider_pagecount) or None,
                    "concurrent_requests": int(concurrent_requests) or None,
                    "log_level": log_level,
                    "llm_backend": llm_backend or None,
                    "llm_endpoint": llm_endpoint or None,
                    "llm_model": llm_model or None,
                    "llm_api_key": llm_api_key or None,
                    "llm_timeout": int(llm_timeout) or None,
                }
                if mode == "Run":
                    resp = post_json("/crawl", params, api_base, int(api_timeout))
                    st.success(f"Run complete. Output: {resp['output_path']}")
                    st.text_area("stdout", resp.get("stdout", ""), height=200)
                    st.text_area("stderr", resp.get("stderr", ""), height=200)
                else:
                    st.info("Dev mode: showing params only")
                    st.json(params)
            else:
                run_payload.update(
                    {
                        "llm_backend": llm_backend or None,
                        "llm_endpoint": llm_endpoint or None,
                        "llm_model": llm_model or None,
                        "llm_api_key": llm_api_key or None,
                        "llm_timeout": int(llm_timeout) or None,
                    }
                )
                resp = post_json("/run", run_payload, api_base, int(api_timeout))
                if mode == "Run":
                    st.success(f"Run complete. Output: {resp['crawl']['output_path']}")
                    st.json(resp.get("params", {}))
                    st.text_area("stdout", resp["crawl"].get("stdout", ""), height=200)
                    st.text_area("stderr", resp["crawl"].get("stderr", ""), height=200)
                else:
                    st.info("Dev mode: interpreted params")
                    st.json(resp.get("params", {}))
        except requests.HTTPError as exc:
            st.error(f"HTTP error: {exc}\n{exc.response.text}")
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.caption("API base can be overridden via API_BASE env. Runs rely on the FastAPI service.")

    st.subheader("Saved runs")
    if st.button("Refresh runs"):
        try:
            runs = requests.get(f"{api_base}/runs-list", timeout=min(20, int(api_timeout))).json()
            if not runs:
                st.info("No runs yet.")
            else:
                for run in runs:
                    size_kb = run["size"] / 1024
                    st.markdown(
                        f"- [{run['name']}]({api_base}{run['url']}) — {size_kb:.1f} KB"
                    )
        except Exception as exc:
            st.error(f"Error fetching runs: {exc}")


if __name__ == "__main__":
    main()

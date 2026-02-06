import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRAPER_WORKDIR = PROJECT_ROOT / "amazon_uk"
SCRAPY_BIN = os.getenv("SCRAPY_BIN", "scrapy")
RUNS_DIR = SCRAPER_WORKDIR / "runs"


class CrawlParams(BaseModel):
    search: str = Field(..., min_length=1, description="Search query text for Amazon")
    category: Optional[str] = Field(default=None, description="Amazon category code (e.g. computers)")
    filter_words: List[str] = Field(default_factory=list, description="Filter words to include")
    filter_mode: Literal["all", "any"] = Field(default="all", description="Match all or any filter words")
    exception_keywords: List[str] = Field(default_factory=list, description="Keywords to exclude")
    closespider_itemcount: Optional[int] = Field(default=None, description="Stop after scraping N items")
    closespider_pagecount: Optional[int] = Field(default=None, description="Stop after crawling N pages")
    concurrent_requests: Optional[int] = Field(default=None, description="Override CONCURRENT_REQUESTS")
    log_level: str = Field(default="INFO", description="Scrapy LOG_LEVEL")
    output: Optional[str] = Field(default=None, description="Path for CSV output; default runs/<timestamp>.csv")


class InterpretRequest(BaseModel):
    prompt: str = Field(..., description="Free-form description of what to search for")
    dev_mode: bool = Field(default=False, description="If true, also return raw LLM text and skip crawl")
    llm_backend: Optional[str] = Field(default=None, description="Override LLM backend (ollama|openai)")
    llm_endpoint: Optional[str] = Field(default=None, description="Override LLM endpoint URL")
    llm_model: Optional[str] = Field(default=None, description="Override LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="Override LLM API key")
    llm_timeout: Optional[float] = Field(default=None, description="Override LLM timeout seconds")


class CrawlResponse(BaseModel):
    success: bool
    output_path: str
    returncode: int
    stdout: str
    stderr: str


class RunRequest(BaseModel):
    prompt: str = Field(..., description="Free-form description of what to search for")
    dev_mode: bool = Field(default=False, description="If true, only interpret; do not crawl")
    llm_backend: Optional[str] = Field(default=None, description="Override LLM backend")
    llm_endpoint: Optional[str] = Field(default=None, description="Override LLM endpoint URL")
    llm_model: Optional[str] = Field(default=None, description="Override LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="Override LLM API key")
    llm_timeout: Optional[float] = Field(default=None, description="Override LLM timeout seconds")
    closespider_itemcount: Optional[int] = Field(default=None, description="Stop after scraping N items")
    closespider_pagecount: Optional[int] = Field(default=None, description="Stop after crawling N pages")
    concurrent_requests: Optional[int] = Field(default=None, description="Override CONCURRENT_REQUESTS")
    log_level: Optional[str] = Field(default=None, description="Override Scrapy LOG_LEVEL")


class LLMClient:
    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "").lower()
        self.endpoint = os.getenv("LLM_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "llama3")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "60"))

    def interpret(
        self,
        prompt: str,
        backend: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> CrawlParams:
        backend = (backend or self.backend).lower()
        endpoint = (endpoint or self.endpoint).rstrip("/") if (endpoint or self.endpoint) else ""
        model = model or self.model
        api_key = api_key or self.api_key
        timeout = timeout or self.timeout

        # If no backend configured, fall back to simple defaults.
        if not backend:
            return CrawlParams(search=prompt)

        try:
            if backend == "ollama":
                return self._interpret_ollama(prompt, endpoint, model, timeout)
            if backend == "openai":
                return self._interpret_openai(prompt, endpoint, model, api_key, timeout)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

        # Unknown backend: use fallback.
        return CrawlParams(search=prompt)

    def _interpret_ollama(self, prompt: str, endpoint: str, model: str, timeout: float) -> CrawlParams:
        if not endpoint:
            raise ValueError("LLM_ENDPOINT required for Ollama")
        payload = {
            "model": model,
            "prompt": self._build_prompt(prompt),
            "stream": False,
        }
        resp = requests.post(f"{endpoint}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        return self._parse_params(text, prompt)

    def _interpret_openai(self, prompt: str, endpoint: str, model: str, api_key: str, timeout: float) -> CrawlParams:
        if not endpoint:
            raise ValueError("LLM_ENDPOINT required for OpenAI backend")
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        resp = requests.post(f"{endpoint}/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        return self._parse_params(text, prompt)

    def _parse_params(self, text: str, fallback_search: str) -> CrawlParams:
        try:
            parsed = self._load_params_from_json(text)
            return parsed
        except Exception as exc:
            snippet = text.strip().replace("\n", " ")[:400]
            print(f"LLM parse error: {exc}; raw (truncated): {snippet}")
            raise HTTPException(status_code=502, detail=f"LLM returned invalid crawl params: {exc}") from exc

    def _load_params_from_json(self, text: str) -> CrawlParams:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")
        params = CrawlParams(**data)
        if not params.search or not params.search.strip():
            raise ValueError("search is empty")
        return params

    def _fallback_params(self, prompt: str) -> CrawlParams:
        return CrawlParams(search=prompt)

    def _build_prompt(self, user_prompt: str) -> str:
        return self._system_prompt() + f"\nUser request: {user_prompt}"

    def _system_prompt(self) -> str:
        return (
            "You convert a short Russian/English shopping request into crawl params JSON for an Amazon search spider.\n"
            "Return ONLY a single JSON object, no prose or markdown. Schema:\n"
            "{\n"
            '  "search": string,               # concise search phrase in English\n'
            '  "category": string|null,        # e.g. "computers", "electronics"\n'
            '  "filter_words": [string],       # include keywords; empty allowed\n'
            '  "filter_mode": "all"|"any",     # default "all"\n'
            '  "exception_keywords": [string], # exclude terms; empty allowed\n'
            '  "closespider_itemcount": int|null,\n'
            '  "closespider_pagecount": int|null,\n'
            '  "concurrent_requests": int|null,\n'
            '  "log_level": "INFO"|"DEBUG",\n'
            '  "output": null                  # API sets filename\n'
            "}\n"
            "Rules: translate to English; put 'без/exclude' terms into exception_keywords; set category when obvious; leave unknowns null/empty; never wrap in code fences.\n"
            "Example input: Найди ноутбуки для ML с 32GB RAM и RTX 4070, без renewed\n"
            'Example output: {"search":"laptop rtx 4070 32gb","category":"computers","filter_words":["rtx 4070","32gb","ml"],"filter_mode":"any","exception_keywords":["renewed","refurbished"],"closespider_itemcount":null,"closespider_pagecount":null,"concurrent_requests":null,"log_level":"INFO","output":null}'
        )


llm_client = LLMClient()
app = FastAPI(title="Amazon UK Scraper Agent", version="0.1.0")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


def _slugify(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    return cleaned.strip("_") or "search"


def build_scrapy_command(params: CrawlParams, output_path: Path) -> List[str]:
    cmd = [SCRAPY_BIN, "crawl", "amazon_uk", "-a", f"search={params.search}"]
    if params.category:
        cmd += ["-a", f"category={params.category}"]
    if params.filter_words:
        cmd += ["-a", f"filter_words={','.join(params.filter_words)}"]
    if params.filter_mode:
        cmd += ["-a", f"filter_mode={params.filter_mode}"]
    if params.exception_keywords:
        cmd += ["-a", f"exception_keywords={','.join(params.exception_keywords)}"]
    cmd += ["-s", f"LOG_LEVEL={params.log_level}"]
    if params.closespider_itemcount:
        cmd += ["-s", f"CLOSESPIDER_ITEMCOUNT={params.closespider_itemcount}"]
    if params.closespider_pagecount:
        cmd += ["-s", f"CLOSESPIDER_PAGECOUNT={params.closespider_pagecount}"]
    if params.concurrent_requests:
        cmd += ["-s", f"CONCURRENT_REQUESTS={params.concurrent_requests}"]
    cmd += ["-o", str(output_path)]
    return cmd


@app.post("/interpret", response_model=CrawlParams)
def interpret(req: InterpretRequest) -> CrawlParams:
    params = llm_client.interpret(
        req.prompt,
        backend=req.llm_backend,
        endpoint=req.llm_endpoint,
        model=req.llm_model,
        api_key=req.llm_api_key,
        timeout=req.llm_timeout,
    )
    return params


@app.post("/crawl", response_model=CrawlResponse)
def crawl(params: CrawlParams) -> CrawlResponse:
    runs_dir = SCRAPER_WORKDIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(params.output) if params.output else runs_dir / f"{_slugify(params.search)}-{int(time.time())}.csv"

    cmd = build_scrapy_command(params, out_path)
    proc = subprocess.run(
        cmd,
        cwd=SCRAPER_WORKDIR,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={"message": "Scrapy crawl failed", "stdout": proc.stdout, "stderr": proc.stderr},
        )

    return CrawlResponse(
        success=True,
        output_path=str(out_path),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


@app.post("/run")
def run(req: RunRequest):
    params = llm_client.interpret(
        req.prompt,
        backend=req.llm_backend,
        endpoint=req.llm_endpoint,
        model=req.llm_model,
        api_key=req.llm_api_key,
        timeout=req.llm_timeout,
    )
    # Optional per-run overrides
    if req.closespider_itemcount is not None:
        params.closespider_itemcount = req.closespider_itemcount
    if req.closespider_pagecount is not None:
        params.closespider_pagecount = req.closespider_pagecount
    if req.concurrent_requests is not None:
        params.concurrent_requests = req.concurrent_requests
    if req.log_level is not None:
        params.log_level = req.log_level
    if req.dev_mode:
        # In dev mode, return only interpreted params for review.
        return {"mode": "dev", "params": params.dict()}

    # Execute crawl with interpreted params
    crawl_resp = crawl(params)
    return {"mode": "run", "params": params.dict(), "crawl": crawl_resp}


@app.get("/health")
def health():
    return {"status": "ok", "backend": llm_client.backend or "none"}


@app.get("/runs-list")
def runs_list():
    if not RUNS_DIR.exists():
        return []
    files = []
    for path in sorted(RUNS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": f"/runs/{path.name}",
            }
        )
    return files

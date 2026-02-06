import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock, Thread
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
    async_mode: bool = Field(default=True, description="If true, run interpret+crawl asynchronously and return a job id")
    debug: bool = Field(default=False, description="Stream crawl stdout/stderr into job status for debugging")


class LLMClient:
    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "").lower()
        self.endpoint = os.getenv("LLM_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "llama3")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "60"))
        self.schema_mode = os.getenv("LLM_SCHEMA_MODE", "auto").lower()  # auto|strict|off

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

        start = time.time()
        print(f"LLM call start: backend={backend} model={model} timeout={timeout}")
        try:
            if backend == "ollama":
                result = self._interpret_ollama(prompt, endpoint, model, timeout)
                print(f"LLM call success: backend={backend} elapsed={time.time() - start:.2f}s")
                return result
            if backend == "openai":
                result = self._interpret_openai(prompt, endpoint, model, api_key, timeout)
                print(f"LLM call success: backend={backend} elapsed={time.time() - start:.2f}s")
                return result
        except requests.Timeout as exc:
            elapsed = time.time() - start
            print(f"LLM call timeout: backend={backend} elapsed={elapsed:.2f}s")
            raise HTTPException(status_code=504, detail=f"LLM call timed out after {elapsed:.2f}s") from exc
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = time.time() - start
            print(f"LLM call failed: backend={backend} elapsed={elapsed:.2f}s error={exc}")
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
        if self.schema_mode != "off":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "crawl_params",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "search": {"type": "string"},
                            "category": {"type": ["string", "null"]},
                            "filter_words": {"type": "array", "items": {"type": "string"}},
                            "filter_mode": {"type": "string", "enum": ["all", "any"]},
                            "exception_keywords": {"type": "array", "items": {"type": "string"}},
                            "closespider_itemcount": {"type": ["integer", "null"]},
                            "closespider_pagecount": {"type": ["integer", "null"]},
                            "concurrent_requests": {"type": ["integer", "null"]},
                            "log_level": {"type": "string", "enum": ["INFO", "DEBUG"]},
                            "output": {"type": ["string", "null"]},
                        },
                        "required": [
                            "search",
                            "category",
                            "filter_words",
                            "filter_mode",
                            "exception_keywords",
                            "closespider_itemcount",
                            "closespider_pagecount",
                            "concurrent_requests",
                            "log_level",
                            "output",
                        ],
                        "additionalProperties": False,
                    },
                },
            }
        url = f"{endpoint}/v1/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            # Some OpenAI-compatible providers (e.g., DeepSeek) don't support response_format=json_schema.
            if self.schema_mode == "auto" and exc.response is not None and exc.response.status_code == 400:
                print("LLM response_format rejected; retrying without schema enforcement.")
                payload.pop("response_format", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
            else:
                raise
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


@dataclass
class JobInfo:
    id: str
    status: str = "pending"  # pending|running|done|error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    prompt: str | None = None
    params: CrawlParams | None = None
    output_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


JOBS: dict[str, JobInfo] = {}
JOBS_LOCK = Lock()


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


def _run_job(job_id: str, req: RunRequest):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = time.time()

    try:
        params = llm_client.interpret(
            req.prompt,
            backend=req.llm_backend,
            endpoint=req.llm_endpoint,
            model=req.llm_model,
            api_key=req.llm_api_key,
            timeout=req.llm_timeout,
        )
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job.params = params

        if req.dev_mode:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    return
                job.status = "done"
                job.finished_at = time.time()
            return

        out_path = RUNS_DIR / f"{_slugify(params.search)}-{int(time.time())}.csv"
        cmd = build_scrapy_command(params, out_path)
        proc = subprocess.Popen(
            cmd,
            cwd=SCRAPER_WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        stderr_lines = []
        if proc.stdout:
            for line in proc.stdout:
                stdout_lines.append(line)
                if req.debug:
                    with JOBS_LOCK:
                        job = JOBS.get(job_id)
                        if job:
                            job.stdout = "".join(stdout_lines)[-4000:]
        if proc.stderr:
            for line in proc.stderr:
                stderr_lines.append(line)
                if req.debug:
                    with JOBS_LOCK:
                        job = JOBS.get(job_id)
                        if job:
                            job.stderr = "".join(stderr_lines)[-4000:]
        returncode = proc.wait()

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job.output_path = str(out_path)
            job.stdout = "".join(stdout_lines)[-4000:]
            job.stderr = "".join(stderr_lines)[-4000:]
            job.finished_at = time.time()
            if returncode != 0:
                job.status = "error"
                job.error = "Scrapy crawl failed"
            else:
                job.status = "done"
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job.status = "error"
            job.error = str(exc)
            job.finished_at = time.time()


def _create_job(prompt: str) -> JobInfo:
    job_id = uuid.uuid4().hex
    job = JobInfo(id=job_id, prompt=prompt)
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


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
    if req.async_mode:
        job = _create_job(req.prompt)
        thread = Thread(target=_run_job, args=(job.id, req), daemon=True)
        thread.start()
        return {"job_id": job.id, "status": job.status}

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


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "id": job.id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "prompt": job.prompt,
            "params": job.params.model_dump() if job.params else None,
            "output_path": job.output_path,
            "stdout": job.stdout,
            "stderr": job.stderr,
            "error": job.error,
        }


@app.get("/jobs")
def jobs_list():
    with JOBS_LOCK:
        return [
            {
                "id": job.id,
                "status": job.status,
                "created_at": job.created_at,
                "prompt": job.prompt,
                "output_path": job.output_path,
                "error": job.error,
            }
            for job in JOBS.values()
        ]


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

# Amazon UK Scrapy Spider

This repository contains a Scrapy spider designed to scrape product information from Amazon UK based on provided search terms, categories, and filters.
It also includes an AI-assisted API/UI that turns free-text requests into crawl parameters before running the scrape.

## Features

- Search products by term and category on Amazon UK.
- Filter results using multiple keywords. For instance, you can search for "Intel NUC" in the "computers" category and filter the results by terms like "i7, 8GB, SSD". The spider supports filtering by multiple comma-separated keywords. 
- **Exception Keywords Filter**: Exclude products containing specific keywords from the scraped results. For example, you can exclude products containing the word "refurbished" from the results.
- Pagination support to scrape multiple pages of search results.
- **Deduplication Filter**: Ensures that the output contains only unique product listings. The deduplication filter inherently removes multiple occurrences of the same sponsored links, ensuring unique listings in the output.
- URL Normalisation: Ensures that all URLs in the output are simplified to remove Amazon-specific tags and SEO-friendly slug.
- Voucher Support: Information about any vouchers available with the product is also included in the results.
- Save results to a CSV file.

## Setup and Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/loglux/Amazon_UK.git
cd Amazon_UK/amazon_uk
```

2. **Set Up a Virtual Environment**:
```bash
python3 -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate.bat instead
``` 

3. **Install Dependencies**:
 ```bash
pip install scrapy
````

4. **Run the Spider**:
```bash
scrapy crawl amazon_uk -a search="Intel NUC" -a category="computers" -a filter_words="i5,i7" -a filter_mode="any" -a exception_keywords="refurbished" -o output.csv
```

## Parameters

- `search`: The search term you want to use (e.g., "Intel NUC").
- `category`: The category within which to search (e.g., "computers").
- `filter_words`: Comma-separated list of words to filter search results. Only results containing all of these words will be returned. Use '-a filter_mode="any"', if you need to change this behaviour. Default is an empty string.
- `filter_mode`: Typically not required, as the default behavior is set to "all", filtering results that contain all specified filter words. However, if you want to change the behavior, set it to "any", which will filter results containing any of the specified filter words.
- `exception_keywords`: Comma-separated list of words that act as negative filters. Results containing any of these words will be excluded. Default is an empty string.

## API wrapper

- Install deps: `pip install -r requirements.txt`.
- Run API (from repo root): `uvicorn api.main:app --reload --port 8000`.
- Docker (API+UI): `docker build -t amazon-uk-scraper .` then `docker run --name amazon-uk-scraper --env-file .env -p 18000:8000 -p 18501:8501 amazon-uk-scraper`.
- Endpoints:
  - `POST /interpret` — optional LLM-backed conversion of free text into crawl params. Configure via env: `LLM_BACKEND=ollama|openai`, `LLM_ENDPOINT`, `LLM_MODEL`, `LLM_API_KEY` (for OpenAI). If unset, falls back to using the prompt as `search`.
    - If the LLM response is invalid JSON, the API returns 502 (no silent fallback).
  - `POST /crawl` — runs the Scrapy spider with JSON params (`search`, `category`, `filter_words`, `filter_mode`, `exception_keywords`, optional `CLOSESPIDER_*`, `CONCURRENT_REQUESTS`, `LOG_LEVEL`, `output`). Returns path to CSV and stdout/stderr.
  - `POST /run` — one-step interpret + crawl; set `dev_mode=true` to only return interpreted params (developer/diagnostic mode).
    - Accepts optional overrides: `closespider_itemcount`, `closespider_pagecount`, `concurrent_requests`, `log_level`.
    - Async by default: returns `job_id`. Check `GET /jobs/{job_id}` for status/results. Set `async_mode=false` to run synchronously.
    - Set `debug=true` to stream crawl stdout/stderr into job status (last ~4KB).
  - `GET /jobs/{job_id}` — check async job status and results.
  - `GET /jobs` — list recent async jobs (in-memory).
  - `GET /health` — basic health info and which LLM backend is active.
- Output CSVs land in `amazon_uk/runs/` by default.
- Offline QA (saved HTML): `python scripts/qa_parse_html.py --html /path/to/search.html --out amazon_uk/runs/qa.csv`
- Online QA (API): `python scripts/qa_online.py --api http://localhost:8000 --pagecount 2`
- LLM timeout can be configured via `LLM_TIMEOUT` (seconds, default 60) if your Ollama/API responses take longer.
 - UI/API timeout in Streamlit can be configured via `API_TIMEOUT` (seconds).

## Streamlit UI

- Start API as above, then run UI from repo root: `streamlit run ui/app.py --server.port 8501`.
- UI supports:
  - Free-text task → interpret (+ crawl if not in dev mode).
  - Manual overrides for search/category/filters/limits.
  - Viewing stdout/stderr and output path.
  - Switching between run/dev (interpret-only) modes.
- LLM config: set envs (or use UI sidebar overrides):
  - Ollama: `LLM_BACKEND=ollama`, `LLM_ENDPOINT=http://<ollama_host>:11434`, `LLM_MODEL=llama3.2`, `LLM_TIMEOUT=300`
  - OpenAI-like: `LLM_BACKEND=openai`, `LLM_ENDPOINT=https://api.deepseek.com` (or your provider), `LLM_MODEL=<model>`, `LLM_API_KEY=<key>`, `LLM_TIMEOUT=120`
  - See `.env.example` for a template; do not commit real keys.

## Default crawler behavior

- Randomizes User-Agent per request from a small desktop/mobile pool.
- AutoThrottle enabled (start delay 2s, max 10s) with base `DOWNLOAD_DELAY=3` to stay polite.
- `ROBOTSTXT_OBEY=False` because Amazon blocks search pages; keep delays as configured.
- Deduplication normalizes `/dp/`, `/gp/product/`, and `sspa`/sponsored links (extracting the ASIN from `url=`) down to `https://www.amazon.co.uk/dp/<ASIN>`.
- Optional flags:
  - `SKIP_SPONSORED=1` (default) drops sponsored blocks early.
  - `ALLOW_EMPTY_PRICE=1` keeps items without a price and marks `missing_price=yes`.
  - `LOG_DROPS=1` writes `runs/drops-<ts>.csv` with drop reason details.
  - `DEBUG_SAVE_EMPTY_HTML=1` saves empty/captcha pages to `runs/empty-<ts>.html`.

## Required Twisted Version

This project was created and tested with a specific version of the Twisted library to ensure compatibility and proper functioning with the Scrapy spider. The required Twisted version for this project is **Twisted 22.10.0**.

### Scrapy Version and Compatibility

At the time this project was created, the latest available version of Scrapy was **Scrapy 2.10.0**. During development and testing, it was confirmed that this version of Scrapy worked seamlessly with Twisted 22.10.0, providing a stable and reliable environment for scraping.

### Compatibility Issue with Newer Twisted Versions

Since software libraries like Scrapy evolve over time, new versions are released to introduce features, improvements, and bug fixes. However, these updates can sometimes lead to compatibility issues with other libraries that the software relies on.

It has been observed that versions of Twisted newer than 22.10.0, such as **Twisted 28.10.0**, can cause compatibility problems with Scrapy 2.10.0. As a result, it is recommended to maintain the specified Twisted version to ensure that the Scrapy spider works as intended.

### Downgrading Twisted for Compatibility

To mitigate the compatibility issue and ensure a smooth experience, it is advised to downgrade Twisted to the required version. You can achieve this by running the following command:

```bash
pip install --upgrade Twisted==22.10.0
```
## Deduplication Filter
To ensure the quality of the scraped data, a deduplication filter is implemented in the Scrapy pipeline. This filter automatically removes any products that have identical or very similar URLs, ensuring that the output contains only unique product listings.

## How it Works
The Scrapy Spider collects product URLs while crawling through the Amazon UK product listings. Before finalizing the scraped data, the pipeline checks each product URL for similarity. If a duplicate or very similar URL is found, that product entry is excluded from the output.

## Usage
The deduplication filter is automatically applied whenever you run the Scrapy spider. Just use the standard command as shown above.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

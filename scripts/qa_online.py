#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen


SAMPLES = [
    {"search": "combo microwave", "category": "appliances"},
    {"search": "intel nuc i7 16gb", "category": "computers"},
    {"search": "refurbished thinkpad", "category": "computers", "exception_keywords": ["refurbished"]},
]


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _map_output_path(path):
    if path.startswith("/app/amazon_uk/runs/"):
        return Path("amazon_uk/runs") / Path(path).name
    return Path(path)


def _summarize_csv(path):
    if not path.exists():
        return {"rows": 0, "missing_price": 0, "sponsored": 0}
    rows = 0
    missing_price = 0
    sponsored = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if row.get("missing_price") == "yes":
                missing_price += 1
            if row.get("sponsored") == "yes":
                sponsored += 1
    return {"rows": rows, "missing_price": missing_price, "sponsored": sponsored}


def main():
    parser = argparse.ArgumentParser(description="Run online QA against the API /crawl.")
    parser.add_argument("--api", default="http://localhost:8000", help="Base API URL.")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout in seconds.")
    parser.add_argument("--pagecount", type=int, default=2, help="CLOSESPIDER_PAGECOUNT value.")
    args = parser.parse_args()

    for sample in SAMPLES:
        payload = dict(sample)
        if isinstance(payload.get("filter_words"), str):
            payload["filter_words"] = [w.strip() for w in payload["filter_words"].split(",") if w.strip()]
        if isinstance(payload.get("exception_keywords"), str):
            payload["exception_keywords"] = [w.strip() for w in payload["exception_keywords"].split(",") if w.strip()]
        payload["closespider_pagecount"] = args.pagecount
        payload["log_level"] = "INFO"
        try:
            result = _post_json(f"{args.api}/crawl", payload, args.timeout)
            output_path = result.get("output_path", "")
            mapped = _map_output_path(output_path) if output_path else None
            summary = _summarize_csv(mapped) if mapped else {"rows": 0, "missing_price": 0, "sponsored": 0}
            print(f"{payload['search']} -> {output_path}")
            print(f"  rows={summary['rows']} missing_price={summary['missing_price']} sponsored={summary['sponsored']}")
        except Exception as exc:
            print(f"{payload['search']} -> ERROR: {exc}")


if __name__ == "__main__":
    main()

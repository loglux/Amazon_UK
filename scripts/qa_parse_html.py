#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from scrapy.http import TextResponse

from amazon_uk.spiders.amazon_uk import AmazonUKSpider
from amazon_uk.pipelines import DuplicatesPipeline
from scrapy.exceptions import DropItem


def main():
    parser = argparse.ArgumentParser(description="Parse a saved Amazon search HTML file using the spider.")
    parser.add_argument("--html", required=True, help="Path to saved HTML file.")
    parser.add_argument("--search", default="Test", help="Search term for logging context.")
    parser.add_argument("--category", default="", help="Category for logging context.")
    parser.add_argument("--filter-words", default="", help="Comma-separated filter words.")
    parser.add_argument("--exception-keywords", default="", help="Comma-separated exception keywords.")
    parser.add_argument("--filter-mode", default="all", choices=["all", "any"], help="Filter mode.")
    parser.add_argument("--out", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    html_path = Path(args.html)
    body = html_path.read_text(encoding="utf-8", errors="ignore")
    response = TextResponse(url="https://www.amazon.co.uk/s?k=test", body=body, encoding="utf-8")

    spider = AmazonUKSpider(
        search=args.search,
        category=args.category,
        filter_words=args.filter_words,
        exception_keywords=args.exception_keywords,
        filter_mode=args.filter_mode,
    )

    pipeline = DuplicatesPipeline()
    items = []
    for item in spider.parse(response):
        try:
            item = pipeline.process_item(item, spider)
            items.append(item)
        except DropItem:
            continue

    print(f"Parsed items: {len(items)}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["asin", "filter", "price", "voucher", "name", "link", "sponsored", "missing_price"],
            )
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()

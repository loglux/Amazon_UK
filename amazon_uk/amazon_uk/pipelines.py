# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import logging
import re
from urllib.parse import parse_qs, unquote, urlparse

from scrapy.exceptions import DropItem

class DuplicatesPipeline:
    def __init__(self):
        self.urls_seen = set()  # Initialize the set to keep track of URLs
        self._asin_pattern = re.compile(r'/(?:dp|gp/product|gp/aw/d|aw/d)/([A-Z0-9]{10})')
        self._asin_only_pattern = re.compile(r'^[A-Z0-9]{10}$')

    def _extract_asin(self, url: str):
        if not url:
            return None
        match = self._asin_pattern.search(url)
        if match:
            return match.group(1)
        # Sponsored links often stash the real URL in a query param.
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            for key in ("url", "u"):
                if key in params and params[key]:
                    candidate = unquote(params[key][0])
                    match = self._asin_pattern.search(candidate)
                    if match:
                        return match.group(1)
            # Try a fully decoded version as a last resort.
            decoded = unquote(url)
            match = self._asin_pattern.search(decoded)
            if match:
                return match.group(1)
        except Exception:
            return None
        return None

    def process_item(self, item, spider):
        # Log the original item link
        logging.info(f"Processing item: {item['link']}")

        # Debugging: Print or log the length of self.urls_seen to see how many unique URLs you have processed
        logging.debug(f"Number of unique URLs seen so far: {len(self.urls_seen)}")

        # Extract the ASIN from the URL using regular expressions
        asin = item.get("asin", "")
        if asin and not self._asin_only_pattern.match(asin):
            asin = ""
        if not asin:
            asin = self._extract_asin(item.get("link", ""))
        if asin:
            purified_url = f"https://www.amazon.co.uk/dp/{asin}"  # The purified URL
            item["asin"] = asin
            item["link"] = purified_url

            # Check if this ASIN has already been seen
            if purified_url in self.urls_seen:
                logging.info(f"Duplicate ASIN dropped: {purified_url}")
                if spider and getattr(spider, "crawler", None):
                    spider.crawler.stats.inc_value("duplicates/dropped", 1)
                if spider and hasattr(spider, "_log_drop"):
                    spider._log_drop(
                        "duplicate_asin",
                        asin=asin,
                        name=item.get("name", ""),
                        price=item.get("price", ""),
                        link=item.get("link", ""),
                        sponsored=item.get("sponsored", "no"),
                        missing_price=item.get("missing_price", "no"),
                    )
                raise DropItem(f"Duplicate item found: {item}")
            else:
                self.urls_seen.add(purified_url)  # Add the new ASIN
                item['link'] = purified_url  # Update the item's link to the purified URL
                if spider and getattr(spider, "crawler", None):
                    spider.crawler.stats.inc_value("duplicates/kept", 1)
                return item  # Return the processed item
        else:
            logging.warning(f"Could not extract ASIN from URL: {item['link']}")
            if spider and getattr(spider, "crawler", None):
                spider.crawler.stats.inc_value("duplicates/invalid_link", 1)
            if spider and hasattr(spider, "_log_drop"):
                spider._log_drop(
                    "invalid_link",
                    asin="",
                    name=item.get("name", ""),
                    price=item.get("price", ""),
                    link=item.get("link", ""),
                    sponsored=item.get("sponsored", "no"),
                    missing_price=item.get("missing_price", "no"),
                )
            raise DropItem(f"Could not extract ASIN: {item}")  # Drop the item if ASIN couldn't be extracted


class AmazonUkPipeline:
    def process_item(self, item, spider):
        return item

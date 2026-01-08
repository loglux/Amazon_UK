# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import logging
import re

from scrapy.exceptions import DropItem

class DuplicatesPipeline:
    def __init__(self):
        self.urls_seen = set()  # Initialize the set to keep track of URLs
        self._asin_pattern = re.compile(r'/(?:dp|gp/product)/([A-Z0-9]{10})')

    def process_item(self, item, spider):
        # Log the original item link
        logging.info(f"Processing item: {item['link']}")

        # Debugging: Print or log the length of self.urls_seen to see how many unique URLs you have processed
        logging.debug(f"Number of unique URLs seen so far: {len(self.urls_seen)}")

        # Extract the ASIN from the URL using regular expressions
        match = self._asin_pattern.search(item['link'])
        if match:
            asin = match.group(1)  # The ASIN
            purified_url = f"https://www.amazon.co.uk/dp/{asin}"  # The purified URL

            # Check if this ASIN has already been seen
            if purified_url in self.urls_seen:
                logging.info(f"Duplicate ASIN dropped: {purified_url}")
                if spider and getattr(spider, "crawler", None):
                    spider.crawler.stats.inc_value("duplicates/dropped", 1)
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
            raise DropItem(f"Could not extract ASIN: {item}")  # Drop the item if ASIN couldn't be extracted


class AmazonUkPipeline:
    def process_item(self, item, spider):
        return item

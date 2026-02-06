import csv
import logging
import os
import time

import scrapy
from ..items import AmazonUkItem


class AmazonUKSpider(scrapy.Spider):
    name = 'amazon_uk'
    allowed_domains = ['www.amazon.co.uk']


    def __init__(self, search="Intel NUC", category="", filter_words="", exception_keywords="", filter_mode="all", *args, **kwargs):
        super(AmazonUKSpider, self).__init__(*args, **kwargs)
        self.search_term = search
        self.category = category
        self.filter_words = [word.strip() for word in filter_words.split(',') if word.strip()] if filter_words else []
        self.exception_keywords = [word.strip() for word in exception_keywords.split(',') if word.strip()] if exception_keywords else []
        self.filter_mode = filter_mode.lower()
        self.debug_save_empty_html = os.getenv("DEBUG_SAVE_EMPTY_HTML", "").lower() in {"1", "true", "yes"}
        self.skip_sponsored = os.getenv("SKIP_SPONSORED", "1").lower() not in {"0", "false", "no"}
        self.log_drops = os.getenv("LOG_DROPS", "").lower() in {"1", "true", "yes"}
        self.drop_log_path = None
        if self.log_drops:
            runs_dir = self._get_runs_dir()
            ts = int(time.time())
            self.drop_log_path = os.path.join(runs_dir, f"drops-{ts}.csv")
            try:
                with open(self.drop_log_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        "reason", "asin", "name", "price", "link", "sponsored",
                        "filter", "missing_price"
                    ])
            except Exception as exc:
                self.logger.warning("Failed to initialize drop log: %s", exc)
                self.drop_log_path = None

        allowed_modes = {"all", "any"}
        if self.filter_mode not in allowed_modes:
            raise ValueError(f"Unsupported filter_mode '{filter_mode}'. Use one of {allowed_modes}.")

    def start_requests(self):
        url = f'https://www.amazon.co.uk/s?k={self.search_term}'
        if self.category:
            url += f'&i={self.category}'
        yield scrapy.Request(url, callback=self.parse)

    def contains_exception_keywords(self, name):
        """Check if the product name contains any exception keyword."""
        return any(keyword.lower() in name.lower() for keyword in self.exception_keywords)

    def contains_filter_words(self, name):
        """Check if the product name contains filter words based on the selected filter_mode."""
        name_cf = name.casefold()
        if not self.filter_words:
            return True
        if self.filter_mode == "any":
            return any(word.casefold() in name_cf for word in self.filter_words)
        # Default to "all" when unrecognized filter_mode values are passed in
        return all(word.casefold() in name_cf for word in self.filter_words)

    def _normalize_text(self, text):
        """Collapse whitespace and strip non-breaking spaces."""
        return " ".join(text.replace("\xa0", " ").split())

    def _extract_name(self, product_element):
        name_parts = product_element.css('span.a-size-base-plus.a-color-base.a-text-normal::text').getall()
        if not name_parts:
            name_parts = product_element.css('span.a-size-medium.a-color-base::text').getall()
        if not name_parts:
            name_parts = product_element.css('h2 a span::text').getall()
        if not name_parts:
            name_parts = product_element.css('h2 span::text').getall()
        if not name_parts:
            name_parts = product_element.css('span.a-size-base-plus.a-color-base::text').getall()
        if not name_parts:
            name_parts = product_element.css('span.a-text-normal::text').getall()
        name = self._normalize_text(''.join(name_parts)) if name_parts else ''
        return name

    def _extract_price(self, product_element):
        price_text = product_element.css('span.a-price .a-offscreen::text').get()
        if price_text:
            return self._normalize_text(price_text)
        price_text = product_element.css('span.a-text-price span.a-offscreen::text').get()
        if price_text:
            return self._normalize_text(price_text)
        price_text = product_element.css('span.a-price[data-a-color="price"] .a-offscreen::text').get()
        if price_text:
            return self._normalize_text(price_text)
        price_text = product_element.css('span.a-price[data-a-size] .a-offscreen::text').get()
        if price_text:
            return self._normalize_text(price_text)
        whole = product_element.css('span.a-price-whole::text').get()
        fraction = product_element.css('span.a-price-fraction::text').get()
        if whole:
            price = whole
            if fraction:
                price = f"{whole}.{fraction}"
            return self._normalize_text(price)
        range_price = product_element.css('span.a-price-range .a-offscreen::text').get()
        if range_price:
            return self._normalize_text(range_price)
        alt_price = product_element.css('span.a-color-price::text').get()
        if alt_price:
            return self._normalize_text(alt_price)
        alt_price = product_element.css('span.a-size-base.a-color-price::text').get()
        if alt_price:
            return self._normalize_text(alt_price)
        # Last resort: any offscreen price-like text within the card.
        candidates = product_element.css("span.a-offscreen::text").getall()
        for text in candidates:
            normalized = self._normalize_text(text)
            if not normalized:
                continue
            lowered = normalized.lower()
            if any(token in lowered for token in ("save", "coupon", "voucher", "rrp")):
                continue
            if "£" in normalized or "€" in normalized or "$" in normalized:
                return normalized
            # Accept common price pattern without currency, e.g. 199.99
            if any(ch.isdigit() for ch in normalized) and "." in normalized:
                return normalized
        return ''

    def _extract_voucher(self, product_element):
        candidates = []
        candidates.extend(
            product_element.css(
                'span.a-size-base.s-highlighted-text-padding.aok-inline-block.s-coupon-highlight-color::text, '
                'span.s-coupon-unclipped span::text, '
                'span.s-coupon-highlight-color::text'
            ).getall()
        )
        candidates.extend(
            product_element.xpath(
                ".//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'coupon') or "
                "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'voucher') or "
                "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'save') or "
                "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'off')]/text()"
            ).getall()
        )

        for text in candidates:
            normalized = self._normalize_text(text)
            if not normalized:
                continue
            lowered = normalized.lower()
            if any(token in lowered for token in ('coupon', 'voucher', 'save', 'off', '£', '%')):
                return normalized
        return ''

    def _is_captcha_page(self, response):
        if response.css("form[action*='validateCaptcha']"):
            return True
        body = response.text.lower()
        return "not a robot" in body or "enter the characters" in body or "validatecaptcha" in body

    def _save_empty_page(self, response):
        if not self.debug_save_empty_html:
            return
        runs_dir = self._get_runs_dir()
        ts = int(time.time())
        fname = f"empty-{ts}.html"
        path = os.path.join(runs_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(response.text)
            self.logger.info("Saved empty page HTML to %s", path)
        except Exception as exc:
            self.logger.warning("Failed to save empty page HTML: %s", exc)

    def _should_keep_without_price(self):
        return os.getenv("ALLOW_EMPTY_PRICE", "").lower() in {"1", "true", "yes"}

    def _get_runs_dir(self):
        runs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "runs")
        runs_dir = os.path.abspath(runs_dir)
        os.makedirs(runs_dir, exist_ok=True)
        return runs_dir

    def _log_drop(self, reason, asin="", name="", price="", link="", sponsored="no", missing_price="no"):
        if not self.drop_log_path:
            return
        try:
            with open(self.drop_log_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    reason, asin, name, price, link, sponsored,
                    ", ".join(self.filter_words) if self.filter_words else "No filter",
                    missing_price
                ])
        except Exception as exc:
            self.logger.debug("Failed to write drop log: %s", exc)

    def _is_sponsored(self, product_element):
        # Skip known sponsored/ad blocks to keep results clean.
        if product_element.xpath(".//*[contains(normalize-space(), 'Sponsored')]"):
            return True
        if product_element.css("[data-component-type^='sp-']"):
            return True
        if product_element.css("[data-ad-id], [data-ad-feedback], [data-asin-sponsered], [data-asin-sponsored]"):
            return True
        if product_element.css("[data-sponsored='true'], [data-ad-marker], [data-sponsored-link]"):
            return True
        if product_element.css(".s-sponsored-label-text, .s-sponsored-label-text-icon"):
            return True
        return False

    def parse(self, response):
        if self._is_captcha_page(response):
            self.logger.warning("Captcha/robot check detected for %s", response.url)
            return
        # Extract product details from the search results page
        product_elements = response.css('div.s-result-item[data-asin]')
        items_found = 0
        for product_element in product_elements:
            asin = product_element.attrib.get('data-asin', '').strip()
            if asin:
                is_sponsored = self._is_sponsored(product_element)
                if self.skip_sponsored and is_sponsored:
                    self._log_drop(
                        "sponsored",
                        asin=asin,
                        name=self._extract_name(product_element),
                        price=self._extract_price(product_element),
                        link="",
                        sponsored="yes"
                    )
                    continue
                name = self._extract_name(product_element)
                price = self._extract_price(product_element)
                voucher = self._extract_voucher(product_element)
                # Extract product link
                link = (
                    product_element.css('h2 a::attr(href)').get()
                    or product_element.css('a.a-link-normal[href*="/dp/"]::attr(href)').get()
                    or product_element.css('a.a-link-normal.s-no-outline::attr(href)').get()
                    or product_element.css('a.a-link-normal::attr(href)').get()
                )
                if link:  # Make the link absolute if it's relative
                    link = response.urljoin(link)
                elif asin:
                    # Fallback to a canonical link when the DOM misses hrefs.
                    link = f"https://www.amazon.co.uk/dp/{asin}"

                if (name and price and
                        self.contains_filter_words(name) and
                        not self.contains_exception_keywords(name)):
                    item = AmazonUkItem()
                    item['asin'] = asin
                    item['filter'] = ', '.join(self.filter_words) if self.filter_words else 'No filter'
                    item['name'] = name
                    item['price'] = price
                    item['voucher'] = voucher if voucher else 'No voucher'
                    item['link'] = link
                    item['sponsored'] = 'yes' if is_sponsored else 'no'
                    item['missing_price'] = 'no'

                    yield item
                    items_found += 1
                else:
                    if not name:
                        self._log_drop("missing_name", asin=asin, name=name, price=price, link=link,
                                       sponsored="yes" if is_sponsored else "no")
                    if not link:
                        self._log_drop("missing_link", asin=asin, name=name, price=price, link=link,
                                       sponsored="yes" if is_sponsored else "no")
                    if name and not self.contains_filter_words(name):
                        self._log_drop("filter_mismatch", asin=asin, name=name, price=price, link=link,
                                       sponsored="yes" if is_sponsored else "no")
                    if name and self.contains_exception_keywords(name):
                        self._log_drop("exception_keyword", asin=asin, name=name, price=price, link=link,
                                       sponsored="yes" if is_sponsored else "no")
                    if self._should_keep_without_price() and name and link and not price:
                        item = AmazonUkItem()
                        item['asin'] = asin
                        item['filter'] = ', '.join(self.filter_words) if self.filter_words else 'No filter'
                        item['name'] = name
                        item['price'] = ''
                        item['voucher'] = voucher if voucher else 'No voucher'
                        item['link'] = link
                        item['sponsored'] = 'yes' if is_sponsored else 'no'
                        item['missing_price'] = 'yes'
                        yield item
                        items_found += 1
                    else:
                        if not name or not price or not link:
                            if not price:
                                self._log_drop("missing_price", asin=asin, name=name, price=price, link=link,
                                               sponsored="yes" if is_sponsored else "no")
                            logging.debug(f"Skipping ASIN {asin}: name='{name}', price='{price}', link='{link}'")
            else:
                continue

        if items_found == 0:
            self.logger.warning("No items parsed on %s (search='%s', category='%s')", response.url, self.search_term, self.category)
            self._save_empty_page(response)

        # Follow pagination links if available
        next_page = response.css('a.s-pagination-next::attr(href)').get() or response.css('li.a-last a::attr(href)').get()
        if next_page:
            yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse)

# Amazon UK Scrapy Spider

This repository contains a Scrapy spider designed to scrape product information from Amazon UK based on provided search terms, categories, and filters.

## Features

- Search products by term and category on Amazon UK.
- Filter results using multiple keywords. For instance, you can search for "Intel NUC" in the "computers" category and filter the results by terms like "i7, 8GB, SSD". The spider supports filtering by multiple comma-separated keywords. 
- **Exception Keywords Filter**: Exclude products containing specific keywords from the scraped results. For example, you can exclude products containing the word "refurbished" from the results.
- Pagination support to scrape multiple pages of search results.
- **Deduplication Filter**: Ensures that the output contains only unique product listings. The deduplication filter inherently removes multiple occurrences of the same sponsored links, ensuring unique listings in the output.
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
scrapy crawl amazon_uk -a search="Intel NUC" -a category="computers" -a filter_words="i5,i7" -a exception_keywords="refurbished" -o output.csv
```

## Parameters

- `search`: The search term you want to use (e.g., "Intel NUC").
- `category`: The category within which to search (e.g., "computers").
- `filter_words`: Comma-separated list of words to filter search results. Only results containing all of these words will be returned. Use '-a filter_mode="any"', if you need to change this behaviour. Default is an empty string.
- `filter_mode`: Typically not required, as the default behavior is set to "all", filtering results that contain all specified filter words. However, if you want to change the behavior, set it to "any", which will filter results containing any of the specified filter words.
- `exception_keywords`: Comma-separated list of words that act as negative filters. Results containing any of these words will be excluded. Default is an empty string.



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


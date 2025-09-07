import scrapy
from scrapy.crawler import CrawlerProcess
import time
import pandas as pd
import os

# Scraping method
class UfcAthleteSpider(scrapy.Spider):
    name = "ufc_athlete"

    # Domain restriction
    allowed_domains = ["ufc.com"]
    # Url to start scraping
    start_urls = ["https://www.ufc.com/athletes/all"]
    # List to store fighter names
    athlete_names = []

    def __init__(self, *args, **kwargs):
        super(UfcAthleteSpider, self).__init__(*args, **kwargs)

    def parse(self, response):
        # Scrape all athlete names on the current page
        for stat in response.css('div.c-listing-athlete-flipcard__action a.e-button--black'):
            href = stat.css('::attr(href)').get()
            if href:
                # Remove "/athlete/" prefix
                slug = href.split("/athlete/")[-1]
                # Convert "conor-mcgregor" → "Conor Mcgregor"
                name = " ".join(word.capitalize() for word in slug.split("-"))
                self.athlete_names.append(name)

        # Check for the Load More button
        next_page = response.css('ul.js-pager__items li.pager__item a::attr(href)').get()
        if next_page:
            next_page_url = response.urljoin(next_page)
            yield scrapy.Request(next_page_url, callback=self.parse)
    
    def close(self, reason):
        # Sort athlete names alphabetically
        self.athlete_names.sort()

        # Save to CSV
        filename = "data/fighters.csv"
        df = pd.DataFrame({'fighter': self.athlete_names})
        df.to_csv(filename, index=False)

        # Get absolute path
        full_path = os.path.abspath(filename)

        print(f"\nTotal UFC Athletes: {len(self.athlete_names)}")
        print(f"\nAthlete names saved to {full_path}")

if __name__ == "__main__":
    start_time = time.time()

    process = CrawlerProcess(settings={
        "USER_AGENT": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 ' '(KHTML, like Gecko) '
            'Chrome/115.0.0.0 Safari/537.36',
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "INFO",
        "RETRY_ENABLED": True,
        "RETRY_HTTP_CODES": [403, 500, 502, 503, 504],
        "DOWNLOAD_TIMEOUT": 15
    })

    process.crawl(UfcAthleteSpider)
    process.start()

    # Calculate scraping duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Scraping Time: {total_time:.2f} seconds")


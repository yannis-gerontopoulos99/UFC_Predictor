import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.item import Item, Field
import logging

class BoutScraperItem(Item):
    event_name = Field()
    event_date = Field()
    fighter_red = Field()
    fighter_blue = Field()
    weight_class = Field()

class Bouts(scrapy.Spider):
    name = 'boutSpider'
    custom_settings = {
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
        }
    }

    def start_requests(self):
        yield scrapy.Request(url='http://ufcstats.com/statistics/events/upcoming', 
                        callback=self.parse,
                        meta={'dont_retry': False})

    def parse(self, response):
        self.logger.info(f"Parsing main page: {response.url}")
        event_anchors = response.css('a.b-link.b-link_style_black')
        self.logger.info(f"Found {len(event_anchors)} potential event links")
        
        event_count = 0
        for anchor in event_anchors:
            event_url = anchor.css('::attr(href)').get()
            event_name = anchor.css('::text').get(default='').strip()
            
            if event_url and event_name:
                event_count += 1
                self.logger.info(f"Queueing event {event_count}: {event_name}")
                yield scrapy.Request(
                    url=event_url,
                    callback=self.parse_event,
                    meta={'event_name': event_name, 'dont_retry': False},
                    errback=self.errback_parse_event
                )

    def parse_event(self, response):
        event_name = response.meta['event_name']
        self.logger.info(f"Parsing event page: {event_name}")
        
        try:
            # Extract date from the list item
            event_date = response.css('li.b-list__box-list-item::text').re_first(r'\w+ \d{1,2}, \d{4}')
            
            if not event_date:
                # self.logger.warning(f"Could not extract date for event: {event_name}")
                event_date = "Unknown"

            # Target the specific data rows you identified
            rows = response.css('tr.b-fight-details__table-row__hover.js-fight-details-click')
            self.logger.info(f"Found {len(rows)} fight rows for event: {event_name}")

            fight_count = 0
            for row in rows:
                try:
                    item = BoutScraperItem()
                    item['event_name'] = event_name
                    item['event_date'] = event_date

                    # Column 2: Fighter Names
                    # We look for the anchor text inside the p tags in the 2nd td
                    fighters = row.css('td:nth-child(2) p a::text').getall()
                    
                    # Column 7: Weight Class
                    # We look for the p tag text in the 7th td
                    weight = row.css('td:nth-child(7) p::text').get()

                    if len(fighters) >= 2:
                        item['fighter_red'] = fighters[0].strip()
                        item['fighter_blue'] = fighters[1].strip()
                        item['weight_class'] = weight.strip() if weight else "Unknown"
                        fight_count += 1
                        yield item
                    else:
                        self.logger.warning(f"Incomplete fighter data in {event_name}: {fighters}")
                        
                except Exception as e:
                    self.logger.error(f"Error parsing row in {event_name}: {e}")
                    continue
            
            self.logger.info(f"Successfully extracted {fight_count} fights from {event_name}\n")
                    
        except Exception as e:
            self.logger.error(f"Error parsing event page {event_name}: {e}\n")

    def errback_parse_event(self, failure):
        request = failure.request
        event_name = request.meta.get('event_name', 'Unknown')
        self.logger.error(f"Failed to fetch event page for {event_name}: {failure.value}")

def scrape_upcoming_events(output_file="data/upcoming_events.csv"):
    """Scrape upcoming UFC events and return the file path."""
    process = CrawlerProcess(settings={
        "FEEDS": {
            output_file: {
                'format': 'csv',
                'encoding': 'utf8',
                'overwrite': True,
                'store_empty': True
            },
        },
        "USER_AGENT": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 ' '(KHTML, like Gecko) '
            'Chrome/115.0.0.0 Safari/537.36',
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "INFO", #Change to DEBUG for more detailed logs
        "RETRY_ENABLED": True,
        "RETRY_HTTP_CODES": [403, 500, 502, 503, 504],
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 30,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 10,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 0.5,
    })

    process.crawl(Bouts)
    process.start() 

    try:
        df = pd.read_csv(output_file)
        if df.empty:
            print(f"Warning: No events were scraped. Check logs above for errors.")
        else:
            # Reorder columns
            df = df[['event_date','event_name','fighter_red','fighter_blue','weight_class']]
            df['event_date'] = pd.to_datetime(df['event_date'], format='%B %d, %Y', errors='coerce')
            df.sort_values(by='event_date', ascending=True, inplace=True)

            upcoiming_event_date = df['event_date'].iloc[0]
            print(f"Upcoming event date: {upcoiming_event_date}")
            # Filter DF to only include rows with the upcoming event date
            df = df[df['event_date'] == upcoiming_event_date]
            
            df.to_csv(output_file, index=False)
            print(f"Success! Saved {len(df)} rows to {output_file}")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty. The scraper did not collect any data. Check the logs above.")
    
    return output_file

if __name__ == "__main__":
    scrape_upcoming_events()

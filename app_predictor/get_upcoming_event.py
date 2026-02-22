import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.item import Item, Field

class BoutScraperItem(Item):
    event_name = Field()
    event_date = Field()
    fighter_red = Field()
    fighter_blue = Field()
    weight_class = Field()

class Bouts(scrapy.Spider):
    name = 'boutSpider'

    def start_requests(self):
        yield scrapy.Request(url='http://ufcstats.com/statistics/events/upcoming', callback=self.parse)

    def parse(self, response):
        event_anchors = response.css('a.b-link.b-link_style_black')
        
        for anchor in event_anchors:
            event_url = anchor.css('::attr(href)').get()
            event_name = anchor.css('::text').get(default='').strip()
            
            if event_url:
                yield scrapy.Request(
                    url=event_url,
                    callback=self.parse_event,
                    meta={'event_name': event_name}
                )

    def parse_event(self, response):
        event_name = response.meta['event_name']
        # Extract date from the list item
        event_date = response.css('li.b-list__box-list-item::text').re_first(r'\w+ \d{1,2}, \d{4}')

        # Target the specific data rows you identified
        rows = response.css('tr.b-fight-details__table-row__hover.js-fight-details-click')

        for row in rows:
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
                yield item

if __name__ == "__main__":
    output_file = "data/upcoming_events.csv"

    process = CrawlerProcess(settings={
        "FEEDS": {
            output_file: {
                'format': 'csv',
                'encoding': 'utf8',
                'overwrite': True,
                'store_empty': True   # ensures headers are written even if no items
            },
        },
        "USER_AGENT": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 ' '(KHTML, like Gecko) '
            'Chrome/115.0.0.0 Safari/537.36',
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "INFO",
        "RETRY_ENABLED": True,
        "RETRY_HTTP_CODES": [403, 500, 502, 503, 504],
        "DOWNLOAD_TIMEOUT": 15
    })

    process.crawl(Bouts)
    process.start() 

    df = pd.read_csv(output_file)
    # Reorder columns
    df = df[['event_date','event_name','fighter_red','fighter_blue','weight_class']]
    df['event_date'] = pd.to_datetime(df['event_date'], format='%B %d, %Y')
    df.sort_values(by='event_date', ascending=True, inplace=True)

    upcoiming_event_date = df['event_date'].iloc[0]
    print(f"Upcoming event date: {upcoiming_event_date}")
    # Filter DF to only include rows with the upcoming event date
    df = df[df['event_date'] == upcoiming_event_date]
    
    df.to_csv(output_file, index=False)
    print(f"Success! Saved to {output_file}")

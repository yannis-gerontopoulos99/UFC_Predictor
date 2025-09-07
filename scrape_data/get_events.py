import datetime
import pandas as pd
import os
import time
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.item import Item, Field
import uuid

# Define fields to extract

class BoutScraperItem(Item):
    event_name = Field()
    event_date = Field()
    fighter_red = Field()
    fighter_blue = Field()
    weight_class = Field()
    win_method = Field()
    #win_method_finish = Field()
    round = Field()
    time = Field()
    winner = Field()
    stance_red = Field()
    stance_blue = Field()
    knockdowns_red = Field()
    knockdowns_blue = Field()
    sig_attempts_red = Field()
    sig_attempts_blue = Field()
    sig_strikes_red = Field()
    sig_strikes_blue = Field()
    total_strikes_attempts_red = Field()
    total_strikes_attempts_blue = Field()
    total_strikes_red = Field()
    total_strikes_blue = Field()
    takedowns_red = Field()
    takedowns_blue = Field()
    takedown_attempts_red = Field()
    takedown_attempts_blue = Field()
    sub_attempts_red = Field()
    sub_attempts_blue = Field()
    control_time_red = Field()
    control_time_blue = Field()
    head_strikes_red = Field()
    head_strikes_blue = Field()
    head_attempts_red = Field()
    head_attempts_blue = Field()
    body_strikes_red = Field()
    body_strikes_blue = Field()
    body_attempts_red = Field()
    body_attempts_blue = Field()
    leg_strikes_red = Field()
    leg_strikes_blue = Field()
    leg_attempts_red = Field()
    leg_attempts_blue = Field()
    distance_red = Field()
    distance_blue = Field()
    distance_attempts_red = Field()
    distance_attempts_blue = Field()
    clinch_strikes_red = Field()
    clinch_strikes_blue = Field()
    clinch_attempts_red = Field()
    clinch_attempts_blue = Field()
    ground_strikes_red = Field()
    ground_strikes_blue = Field()
    ground_attempts_red = Field()
    ground_attempts_blue = Field()

# Scraping method
class Bouts(scrapy.Spider):
    name = 'boutSpider'

    def start_requests(self):
        # Start scraping from the ufc completed events page
        start_urls = [
            'http://ufcstats.com/statistics/events/completed?page=all'
        ]
        # Issue requests to all start urls
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Get all event blocks (name and link are in the same <a>)
        event_anchors = response.css('a.b-link.b-link_style_black')
        
        for anchor in event_anchors:
            event_url = anchor.css('::attr(href)').get()
            event_name = anchor.css('::text').get(default='').strip()
            
            # Only proceed if both url and name exist
            if event_url and event_name:
                yield scrapy.Request(
                    url=event_url,
                    callback=self.parse_event,
                    meta={'event_name': event_name}
                )

    def parse_event(self, response):
        # Get event name from meta
        event_name = response.meta['event_name']

        # Extract event date from the list item (e.g., "Date: July 31, 2025"
        event_date = response.css('li.b-list__box-list-item::text').re_first(r'\w+ \d{1,2}, \d{4}')
        
        if not event_date:
            self.logger.warning(f"Event date not found for: {event_name}")

        # Extract all fight detail page links from the event page
        fight_links = response.css('td.b-fight-details__table-col.b-fight-details__table-col_style_align-top a::attr(href)').getall()
        
        for fight_url in fight_links:
            # Request each fight url, passing event info
            yield scrapy.Request(
                url=fight_url,
                callback=self.parse_fight,
                meta={'event_name': event_name, 'event_date': event_date}
            )

    def parse_fight(self, response):
        # Extract ALL data first (basic info + stats)
        item = self.extract_all_fight_data(response)
        
        # Initialize temp_items at class level if it doesn't exist
        if not hasattr(self, 'temp_items'):
            self.temp_items = {}
        
        # Get fighter profile URLs for stances
        fighter_anchors = response.css('tbody.b-fight-details__table-body tr.b-fight-details__table-row td.b-fight-details__table-col a')
        if len(fighter_anchors) < 2:
            self.logger.warning("Could not find fighter URLs")
            yield item  # Yield without stances
            return
        
        red_url = fighter_anchors[0].css('::attr(href)').get()
        blue_url = fighter_anchors[1].css('::attr(href)').get()
        
        # If no URLs, yield item immediately
        if not red_url and not blue_url:
            yield item
            return
        
        # Store item and fetch stances
        fight_id = str(uuid.uuid4())
        
        # Initialize the fight data
        self.temp_items[fight_id] = {
            'item': item,
            'stances_needed': 0,
            'stances_collected': 0
        }
        
        # Count how many stances we need and make requests
        if red_url:
            self.temp_items[fight_id]['stances_needed'] += 1
            yield scrapy.Request(
                url=red_url,
                callback=self.parse_stance,
                meta={'fight_id': fight_id, 'color': 'red'},
                dont_filter=True  # Ensure request is made even if URL was visited
            )
        
        if blue_url:
            self.temp_items[fight_id]['stances_needed'] += 1
            yield scrapy.Request(
                url=blue_url,
                callback=self.parse_stance,
                meta={'fight_id': fight_id, 'color': 'blue'},
                dont_filter=True  # Ensure request is made even if URL was visited
            )
        
        # If no stance URLs found, yield immediately
        if self.temp_items[fight_id]['stances_needed'] == 0:
            yield item
            del self.temp_items[fight_id]

    def parse_stance(self, response):
        """Extract stance and yield item when all stances collected"""
        fight_id = response.meta['fight_id']
        color = response.meta['color']
        
        # Initialize temp_items if it doesn't exist (defensive programming)
        if not hasattr(self, 'temp_items'):
            self.temp_items = {}
            self.logger.error(f"temp_items not initialized for fight_id: {fight_id}")
            return
        
        # Check if fight_id exists
        if fight_id not in self.temp_items:
            self.logger.error(f"Fight ID {fight_id} not found in temp_items")
            return
        
        fight_data = self.temp_items[fight_id]
        item = fight_data['item']
        
        # Extract stance
        stance_text = response.xpath(
            'normalize-space(string(//li[contains(@class,"b-list__box-list-item")][i[contains(text(),"STANCE:")]]))'
        ).get()
        
        stance = stance_text.replace("STANCE:", "").strip() if stance_text else None
        
        # Set stance
        item[f'stance_{color}'] = stance
        
        # Update counter
        fight_data['stances_collected'] += 1
        
        self.logger.debug(f"Fight {fight_id}: collected {fight_data['stances_collected']}/{fight_data['stances_needed']} stances")
        
        # YIELD when all stances collected
        if fight_data['stances_collected'] >= fight_data['stances_needed']:
            self.logger.debug(f"Yielding complete item for fight: {item.get('fighter_red')} vs {item.get('fighter_blue')}")
            yield item
            del self.temp_items[fight_id]
        
    def extract_all_fight_data(self, response):
        """Extract all fight data (basic info + stats) in one place"""
        item = BoutScraperItem()
        
        # Basic metadata
        item['event_name'] = response.meta['event_name']
        item['event_date'] = response.meta['event_date']

        # Extract basic fight info
        item['win_method'] = response.css('.b-fight-details__text-item_first').xpath(
            ".//i[contains(text(), 'Method:')]/following-sibling::i/text()"
        ).get(default='').strip()

        # Round and Time
        details = response.css('div.b-fight-details__content i.b-fight-details__text-item')
        for detail in details:
            label = detail.css('i.b-fight-details__label::text').get()
            if label:
                label = label.strip()
                if label == "Round:":
                    item['round'] = detail.css('::text').re_first(r'\d+')
                elif label == "Time:":
                    time_text = detail.css('::text').re_first(r'\d+:\d+')
                    if time_text:
                        minutes, seconds = map(int, time_text.split(':'))
                        item['time'] = int(datetime.timedelta(minutes=minutes, seconds=seconds).total_seconds())

        # Weight class
        weight_class = response.xpath("string(//i[@class='b-fight-details__fight-title'])").get()
        if weight_class:
            weight_class = weight_class.strip().replace(' Bout', '')
        item['weight_class'] = weight_class
        
        # Fighter names
        fighter_anchors = response.css('tbody.b-fight-details__table-body tr.b-fight-details__table-row td.b-fight-details__table-col a')
        if len(fighter_anchors) >= 2:
            item['fighter_red'] = fighter_anchors[0].css('::text').get(default='').strip()
            item['fighter_blue'] = fighter_anchors[1].css('::text').get(default='').strip()

        # Initialize stances
        item['stance_red'] = None
        item['stance_blue'] = None

        # Extract all fight stats
        self.extract_fight_stats(response, item)
        
        return item

    def extract_fight_stats(self, response, item):
        """Extract fight statistics"""
        
        def extract_pair(col_index, split=False, table_index=0):
            try:
                tables = response.css('tbody.b-fight-details__table-body')
                if len(tables) <= table_index:
                    return (None,) * (4 if split else 2)

                cols = tables[table_index].css('td.b-fight-details__table-col')
                vals = cols[col_index].css('p::text').getall()

                if split:
                    red = vals[0].strip().split(' of ')
                    blue = vals[1].strip().split(' of ')
                    return red[0], red[1], blue[0], blue[1]
                else:
                    return vals[0].strip(), vals[1].strip()
            except:
                return (None,) * (4 if split else 2)

        def mmss_to_seconds(time_str):
            try:
                if time_str and time_str.strip():
                    minutes, seconds = map(int, time_str.strip().split(':'))
                    return minutes * 60 + seconds
                return 0
            except Exception:
                return 0

        # Table 0
        # Knockdowns (col 1)
        item['knockdowns_red'], item['knockdowns_blue'] = extract_pair(1)
        
        # Significant strikes (col 2)
        sr, sa_r, sb, sa_b = extract_pair(2, split=True)
        item['sig_strikes_red'] = sr
        item['sig_attempts_red'] = sa_r
        item['sig_strikes_blue'] = sb
        item['sig_attempts_blue'] = sa_b

        # Total strikes (col 4)
        tr, ta_r, tb, ta_b = extract_pair(4, split=True)
        item['total_strikes_red'] = tr
        item['total_strikes_attempts_red'] = ta_r
        item['total_strikes_blue'] = tb
        item['total_strikes_attempts_blue'] = ta_b

        # Takedowns (col 5)
        td_r, td_att_r, td_b, td_att_b = extract_pair(5, split=True)
        item['takedowns_red'] = td_r
        item['takedown_attempts_red'] = td_att_r
        item['takedowns_blue'] = td_b
        item['takedown_attempts_blue'] = td_att_b

        # Submission attempts (col 7)
        item['sub_attempts_red'], item['sub_attempts_blue'] = extract_pair(7)
        
        # Control time (col 9)
        ctrl_red, ctrl_blue = extract_pair(9)
        item['control_time_red'] = mmss_to_seconds(ctrl_red)
        item['control_time_blue'] = mmss_to_seconds(ctrl_blue)

        # Table 3
        # Head strikes (col 3)
        hr, ha_r, hb, ha_b = extract_pair(3, split=True, table_index=2)
        item['head_strikes_red'] = hr
        item['head_attempts_red'] = ha_r
        item['head_strikes_blue'] = hb
        item['head_attempts_blue'] = ha_b

        # Body strikes (col 4)
        br, ba_r, bb, ba_b = extract_pair(4, split=True, table_index=2)
        item['body_strikes_red'] = br
        item['body_attempts_red'] = ba_r
        item['body_strikes_blue'] = bb
        item['body_attempts_blue'] = ba_b

        # Leg strikes (col 5)
        lr, la_r, lb, la_b = extract_pair(5, split=True, table_index=2)
        item['leg_strikes_red'] = lr
        item['leg_attempts_red'] = la_r
        item['leg_strikes_blue'] = lb
        item['leg_attempts_blue'] = la_b

        # Distance strikes (col 6)
        dr, da_r, db, da_b = extract_pair(6, split=True, table_index=2)
        item['distance_red'] = dr
        item['distance_attempts_red'] = da_r
        item['distance_blue'] = db
        item['distance_attempts_blue'] = da_b

        # Clinch strikes (col 7)
        cr, ca_r, cb, ca_b = extract_pair(7, split=True, table_index=2)
        item['clinch_strikes_red'] = cr
        item['clinch_attempts_red'] = ca_r
        item['clinch_strikes_blue'] = cb
        item['clinch_attempts_blue'] = ca_b

        # Ground strikes (col 8)
        gr, ga_r, gb, ga_b = extract_pair(8, split=True, table_index=2)
        item['ground_strikes_red'] = gr
        item['ground_attempts_red'] = ga_r
        item['ground_strikes_blue'] = gb
        item['ground_attempts_blue'] = ga_b

        # Winner
        winner_block = response.css('div.b-fight-details__person:has(i.b-fight-details__person-status_style_green)')
        winner_name = winner_block.css('h3.b-fight-details__person-name a::text').get()
        item['winner'] = winner_name.strip() if winner_name else None

if __name__ == "__main__":
    
    output_file = "data/events.csv"
    start_time = time.time()

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

    # Start process
    process.crawl(Bouts)
    process.start()
    
    end_time = time.time()
    print(f"Scraping completed in {end_time - start_time:.2f} seconds.")
    print("Data scraped and saved!")

    # Sort by date and change column order
    df = pd.read_csv(output_file)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df_sorted = df.sort_values(by='event_date', ascending=True)
    df_sorted = df_sorted.reindex(columns=['event_date','event_name','fighter_red','fighter_blue','round','time','weight_class',
        'win_method','winner','stance_red', 'stance_blue','knockdowns_red','knockdowns_blue','sig_attempts_red','sig_attempts_blue',
        'sig_strikes_red','sig_strikes_blue','total_strikes_attempts_red','total_strikes_attempts_blue','total_strikes_red',
        'total_strikes_blue','sub_attempts_red','sub_attempts_blue','takedowns_red','takedowns_blue','takedown_attempts_red',
        'takedown_attempts_blue','control_time_red','control_time_blue','head_strikes_red','head_strikes_blue','head_attempts_red',
        'head_attempts_blue','body_strikes_red','body_strikes_blue','body_attempts_red','body_attempts_blue','leg_strikes_red',
        'leg_strikes_blue','leg_attempts_red','leg_attempts_blue','distance_red','distance_blue','distance_attempts_red',
        'distance_attempts_blue','clinch_strikes_red','clinch_strikes_blue','clinch_attempts_red','clinch_attempts_blue',
        'ground_strikes_red','ground_strikes_blue','ground_attempts_red','ground_attempts_blue'])
    df_sorted.to_csv(output_file, index=False)

    print(f"Data saved and sorted to: {os.path.abspath(output_file)}")

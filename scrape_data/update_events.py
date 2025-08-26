from __future__ import absolute_import
import datetime
import pandas as pd
import os
from dotenv import load_dotenv
import time
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.item import Item, Field
import subprocess
import mysql.connector

load_dotenv()  # load variables from .env file

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
    knockdowns_red = Field()
    knockdowns_blue = Field()
    sig_strikes_red = Field()
    sig_strikes_blue = Field()
    sig_attempts_red = Field()
    sig_attempts_blue = Field()
    total_strikes_red = Field()
    total_strikes_blue = Field()
    total_strikes_attempts_red = Field()
    total_strikes_attempts_blue = Field()
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
        # Create an item to hold extracted bout data
        item = BoutScraperItem()

        # Set basic metadata from previous callbacks
        item['event_name'] = response.meta['event_name']
        item['event_date'] = response.meta['event_date']

        # Extract win method
        item['win_method'] = response.css('.b-fight-details__text-item_first').xpath(
            ".//i[contains(text(), 'Method:')]/following-sibling::i/text()"
        ).get(default='').strip()

        # Extract Round and Time
        details = response.css('div.b-fight-details__content i.b-fight-details__text-item')

        for detail in details:
            label = detail.css('i.b-fight-details__label::text').get()
            if label:
                label = label.strip()
                if label == "Round:":
                    round_text = detail.css('::text').re_first(r'\d+')
                    item['round'] = round_text
                elif label == "Time:":
                    time_text = detail.css('::text').re_first(r'\d+:\d+')
                    minutes, seconds = map(int, time_text.split(':'))
                    item['time'] = time_text
                    item['time'] = int(datetime.timedelta(minutes=minutes, seconds=seconds).total_seconds())

        # Extract weight class (e.g., Lightweight Bout)
        weight_class = response.xpath("string(//i[@class='b-fight-details__fight-title'])").get()
        if weight_class:
            # Clean up trailing " Bout"
            weight_class = weight_class.strip().replace(' Bout', '')
        item['weight_class'] = weight_class
        
        # Get table row for fighter total stats
        row = response.css('tbody.b-fight-details__table-body tr.b-fight-details__table-row')
        if not row:
            self.logger.warning("No stats row found.")
            return
        
        # Get each column
        cols = row.css('td.b-fight-details__table-col')

        # Fighter names (column 0)
        fighter_names = cols[0].css('a::text').getall()
        if len(fighter_names) >= 2:
            item['fighter_red'] = fighter_names[0].strip()
            item['fighter_blue'] = fighter_names[1].strip()
        
        # Helper function to extract red/blue fighter stats from a column
        def extract_pair(col_index, split=False, table_index=0):
            """
            Extract red/blue stats from a given column index.
            
            table_index: which stats table to use (0 = first table, 1 = second table)
            """
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

        # Convert MM:SS format to seconds
        def mmss_to_seconds(time_str):
            try:
                minutes, seconds = map(int, time_str.strip().split(':'))
                return minutes * 60 + seconds
            except Exception:
                return 0  # Or None
        
        item['control_time_red'], item['control_time_blue'] = mmss_to_seconds(ctrl_red), mmss_to_seconds(ctrl_blue)

        # Extract winner's name where 'W' is green
        winner_block = response.css('div.b-fight-details__person:has(i.b-fight-details__person-status_style_green)')
        winner_name = winner_block.css('h3.b-fight-details__person-name a::text').get()
        if winner_name:
            item['winner'] = winner_name.strip()

        # Return bout items
        yield item

if __name__ == "__main__":

    #output_file = r'data/events.csv'
    start_time = time.time()

    # Load data from csv
    '''
    # Load existing file and get latest event date
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_df['event_date'] = pd.to_datetime(existing_df['event_date'], errors='coerce')
        latest_date = existing_df['event_date'].max()
    else:
        existing_df = pd.DataFrame()
        latest_date = None
    '''

    # Load data from MySQL
    conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    database=os.getenv("DB_NAME"),
    password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()
    query = ("SELECT MAX(event_date) FROM events")
    cursor.execute(query)
    latest_date = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    if latest_date:
        print("Max date found:", latest_date)
    else:
        latest_date = None
        print('Warning: No event_date found in the table.')

    # Create temporarily file to scrape newer dates
    temp_file = 'data/temp_bouts.csv'

    process = CrawlerProcess(settings={
        "FEEDS": {
            temp_file: {
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

    # Pass latest_date as string to avoid serialization issues
    latest_date_ts = pd.Timestamp(latest_date) if latest_date is not None else None
    print(latest_date_ts)

    # Start process
    process.crawl(Bouts, latest_event_date=latest_date_ts)
    process.start()

    print("Scraping done!")
    end_time = time.time()
    print(f"Total scraping and processing time: {end_time - start_time:.2f} seconds.")

    correct_order = [
        'event_date','event_name','fighter_blue','fighter_red','round','time','weight_class','win_method','winner',
        'knockdowns_blue','knockdowns_red','sig_attempts_blue','sig_attempts_red','sig_strikes_blue','sig_strikes_red',
        'total_strikes_attempts_blue','total_strikes_attempts_red','total_strikes_blue','total_strikes_red','sub_attempts_blue',
        'sub_attempts_red','takedowns_blue','takedowns_red','takedown_attempts_blue','takedown_attempts_red','control_time_blue',
        'control_time_red','head_strikes_red','head_strikes_blue','head_attempts_red','head_attempts_blue','body_strikes_red',
        'body_strikes_blue','body_attempts_red','body_attempts_blue','leg_strikes_red','leg_strikes_blue','leg_attempts_red',
        'leg_attempts_blue','distance_red','distance_blue','distance_attempts_red','distance_attempts_blue','clinch_strikes_red',
        'clinch_strikes_blue','clinch_attempts_red','clinch_attempts_blue','ground_strikes_red','ground_strikes_blue',
        'ground_attempts_red','ground_attempts_blue']
    
    # Load the CSV into a DataFrame
    temp_df = pd.read_csv(temp_file)
    # Reorder the columns
    temp_df = temp_df.reindex(columns=correct_order)
    # Save it back to the same file
    temp_df.to_csv(temp_file, index=False)

    if os.path.exists(temp_file):
        new_df = pd.read_csv(temp_file)
        new_df['event_date'] = pd.to_datetime(new_df['event_date'], errors='coerce')

        # Filter only new rows (by event_date > latest)
        if latest_date_ts is not None:
            temp_filtered_df = new_df[new_df['event_date'] > latest_date_ts].copy()
        else:
            temp_filtered_df = new_df.copy()

        # Overwrite temp_file with only new rows
        temp_filtered_df.to_csv(temp_file, index=False)

        if len(temp_filtered_df) > 0:
            # Use csv
            '''
            # Merge into master file
            combined_df = pd.concat([existing_df, temp_filtered_df], ignore_index=True)
            combined_df.drop_duplicates(inplace=True)
            combined_df = combined_df.sort_values(by='event_date', ascending=True)
            combined_df = combined_df.reindex(columns=['event_date','event_name','fighter_blue','fighter_red','round','time','weight_class',
            'win_method','winner','knockdowns_blue','knockdowns_red','sig_attempts_blue','sig_attempts_red','sig_strikes_blue','sig_strikes_red',
            'total_strikes_attempts_blue','total_strikes_attempts_red','total_strikes_blue','total_strikes_red','sub_attempts_blue',
            'sub_attempts_red','takedowns_blue','takedowns_red','takedown_attempts_blue','takedown_attempts_red','control_time_blue',
            'control_time_red','head_strikes_red','head_strikes_blue','head_attempts_red','head_attempts_blue','body_strikes_red',
            'body_strikes_blue','body_attempts_red','body_attempts_blue','leg_strikes_red','leg_strikes_blue','leg_attempts_red',
            'leg_attempts_blue','distance_red','distance_blue','distance_attempts_red','distance_attempts_blue','clinch_strikes_red',
            'clinch_strikes_blue','clinch_attempts_red','clinch_attempts_blue','ground_strikes_red','ground_strikes_blue',
            'ground_attempts_red','ground_attempts_blue'])
            combined_df.to_csv(output_file, index=False)
            print(f"Updated data saved to: {os.path.abspath(output_file)}")
            print("Number of new fights added: ", len(temp_filtered_df))
            '''

            # Run script to INSERT INTO new events to DB
            subprocess.run(['python', 'DB_connections/append_events_DB.py'], check=True)
            # Run script to UPDATE stats of the fighters in the new events DB
            subprocess.run(['python', 'scrape_date/update_stats.py'], check=True)
        else:
            print("No new entries found.")
    else:
        print("No temp file found.")
    
    # Remove temporarily file
    #os.remove(temp_file)
    
import scrapy
from scrapy.crawler import CrawlerProcess
import time
import pandas as pd
import os
from dotenv import load_dotenv
import subprocess
import csv
import mysql.connector

load_dotenv()  # load variables from .env file

# Scraping mehtod
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
        for stat in response.css('div.c-listing-athlete__text'):
            name = stat.css('span.c-listing-athlete__name::text').get()
            if name:
                name = name.strip()
                self.athlete_names.append(name)
    
        # Check for the Load More button
        next_page = response.css('ul.js-pager__items li.pager__item a::attr(href)').get()
        if next_page:
            next_page_url = response.urljoin(next_page)
            yield scrapy.Request(next_page_url, callback=self.parse)
    
    def close(self):
        # Sort athlete names alphabetically
        self.athlete_names.sort()

        # Main file
        #old_file = "data/fighters.csv"

        # Temporarily file
        temp_fighters = "data/temp_fighters.csv"
        with open(temp_fighters, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([['fighter']])

        #Load from csv
        '''
        # Load existing data if available into a set
        if os.path.exists(old_file):
            existing_df = pd.read_csv(old_file)
            existing_names = set(existing_df['fighter'].dropna().str.strip())
        else:
            existing_names = set()
        '''

        import mysql.connector
        # Event data
        conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        database=os.getenv("DB_NAME"),
        password=os.getenv("DB_PASSWORD")
        )
        cursor = conn.cursor()
        query = ("SELECT fighter FROM fighters")
        cursor.execute(query)
        existing_names = cursor.fetchall()

        cursor.close()
        conn.close()

        existing_names = set(name[0].strip() for name in existing_names if name[0] is not None)


        # Find new_names from removing old ones
        new_names = sorted(set(self.athlete_names) - existing_names)

        if new_names:
            # Load new_names to temp_file
            temp_df = pd.DataFrame({'fighter': new_names})
            temp_df.to_csv(temp_fighters, index=False)
            print(f"\nNew fighter names saved to {os.path.abspath(temp_fighters)}")
            print("New fighters added: ", len(temp_df))

            '''
            # Save all_names to main file
            all_names = existing_names.union(self.athlete_names)
            final_df = pd.DataFrame({'fighter': sorted(all_names)})
            final_df = final_df.sort_values(by='fighter', ascending=True, key=lambda col: col.str.lower())
            final_df.to_csv(old_file, index=False)
            '''

            # Run script to INSERT INTO new fighters to DB
            subprocess.run(['python', 'DB_connections/append_fighters_DB.py', temp_fighters], check=True)
            # Run script to INSERT INTO new stats to DB
            subprocess.run(['python', 'DB_connections/append_stats.py', temp_fighters], check=True) # Append CSV and DB

            # Retrive total fighters
            conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            database=os.getenv("DB_NAME"),
            password=os.getenv("DB_PASSWORD")
            )
            cursor = conn.cursor()
            query = ("SELECT COUNT(fighter) FROM fighters")
            cursor.execute(query)
            total_fighters = cursor.fetchone()[0]

            cursor.close()
            conn.close()


            print("\nTotal UFC Athletes: ", total_fighters)
            #print(f"File updated in {os.path.abspath(old_file)}")

        else:
            print("\nNo new entries found.")
            #print(f"Athlete names unchanged at {os.path.abspath(old_file)}")

        # Remove temporarily file
        os.remove(temp_fighters)


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

    # Calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Scraping Time: {total_time:.2f} seconds")
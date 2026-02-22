import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd
import re
from datetime import datetime

from batch_predictor import UFCPredictionPipeline
from get_upcoming_event import Bouts
from scrapy.crawler import CrawlerProcess

def scrape_upcoming_event():
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
    
    return output_file

def mysql_conn(event_date):
    load_dotenv()
    db_config = {
        'host': os.getenv("DB_HOST"),
        'user': os.getenv("DB_USER"), 
        'database': os.getenv("DB_NAME"),
        'password': os.getenv("DB_PASSWORD")
    }

    # 1. Unpack config with **
    conn = mysql.connector.connect(**db_config)
    
    # 2. Use a parameterized query to prevent SQL errors and injections
    query = """
        SELECT event_date, event_name, fighter_red, fighter_blue, weight_class, winner 
        FROM events 
        WHERE event_date >= %s 
        ORDER BY event_date DESC
    """
    
    # 3. Use Pandas to read directly (much cleaner than manual fetching)
    df_events = pd.read_sql(query, conn, params=(event_date,))
    
    conn.close()
    return df_events

def get_max_date():
    load_dotenv()
    db_config = {
        'host': os.getenv("DB_HOST"),
        'user': os.getenv("DB_USER"), 
        'database': os.getenv("DB_NAME"),
        'password': os.getenv("DB_PASSWORD")
    }

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    # SQL query to find the single most recent date
    query = "SELECT MAX(event_date) FROM events"
    
    cursor.execute(query)
    # fetchone() returns a tuple like (datetime.date(2026, 1, 31),)
    result = cursor.fetchone()
    
    conn.close()

    # result[0] will be None if the table is empty
    if result and result[0]:
        return result[0]
    return None

def get_clean_date():
    """Prompts user for date, removes non-numeric characters except hyphens, and validates."""
    while True:
        user_input = input("\nEnter start date (YYYY-MM-DD) or 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            return 'q'
            
        # 1. Clean the string: Remove everything except digits and hyphens
        # This fixes accidental inputs like "date: 2025-12-13!"
        normalized = re.sub(r'[./ ]', '-', user_input)
        clean_date = re.sub(r'[^0-9-]', '', normalized)
        
        # 2. Validate format using datetime
        try:
            valid_date = datetime.strptime(clean_date, '%Y-%m-%d')
            return clean_date # Returns the cleaned string
        except ValueError:
            print(f"Invalid format: '{user_input}'. Please use YYYY-MM-DD (e.g., 2025-12-13).")

def main():
    pipeline = UFCPredictionPipeline()
    #output_path = "data/predictions_upcoming.csv"

    while True:
        print("\n" + "="*30)
        print("UFC PREDICTOR MENU")
        print("="*30)
        print("1. Predict the next upcoming event (Scrape)")
        print("2. Predict from a specific date onwards (MySQL)")
        print("3. Quit")
        
        choice = input("\nChoice: ").strip().lower()

        if choice == '1':
            print("Scraping upcoming event data...")
            output_path = "data/predictions_upcoming.csv"
            file_path = scrape_upcoming_event()
            # Pass the file path to the pipeline
            pipeline.run_batch_pipeline(file_path, output_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            print(f"Predictions saved to {output_path}")

        elif choice == '2':
            latest_date = get_max_date()
            if latest_date:
                print(f"The most recent fight in the database is from: {latest_date}")
                date_val = get_clean_date()
                if date_val == 'q' or date_val == '3': break
                
                print(f"🔍 Fetching fights from MySQL starting from {date_val}...")
                df_fights = mysql_conn(date_val)
                
                if df_fights.empty:
                    print("No fights found in the database for that date range.")
                else:
                    # Pass the DATAFRAME directly to the pipeline
                    output_path = "data/predictions_older.csv"
                    pipeline.run_batch_pipeline(df_fights, output_path)
                    print(f"Predictions saved to {output_path}")
            else:
                print("The database is currently empty.")

        elif choice == 'q' or choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please type '1', '2', or '3'.")

if __name__ == "__main__":
    main()

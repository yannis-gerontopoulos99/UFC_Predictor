import pandas as pd
import os
import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
import time
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()  # load variables from .env file

# Extract fighters and convert into one column
def extract_fighters(temp_file='data/temp_bouts.csv'):
    if not os.path.exists(temp_file):
        print(f"File not found: {temp_file}")
        return []
    
    df = pd.read_csv(temp_file)

    if 'fighter_blue' not in df.columns or 'fighter_red' not in df.columns:
        print("Columns 'fighter_blue' and/or 'fighter_red not found in the file")
        return []
    
    # Concat two columns together
    fighter_series = pd.concat([df['fighter_blue'], df['fighter_red']], ignore_index=True)
    fighter_series = fighter_series.dropna().astype(str)

    fighter_df = pd.DataFrame({'fighter': fighter_series})

    return fighter_df

# Track failed URLs and reasons
failed_urls = []
successful_names = []

# Scraping method
class UfcAthleteSpider(scrapy.Spider):
    name = "ufc_athlete"
    # Domain name
    allowed_domains = ["ufc.com"]

    def __init__(self, athlete_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.athlete_names = athlete_names or []

    def start_requests(self):
        # Url to start scraping from using the fighter names
        for name in self.athlete_names:
            url = f"https://www.ufc.com/athlete/{name}"
            yield scrapy.Request(url=url, callback=self.parse)
            
    def parse(self, response):
            # Extract fighter name from URL for debugging
            url_fighter_name = response.url.split('/')[-1]
            
            # Check if we got a 404 or redirect
            if response.status == 404:
                self.logger.warning(f"404 Error for fighter: {url_fighter_name}")
                failed_urls.append((url_fighter_name, "404_not_found"))
                return
                
            # Check if we were redirected (might indicate fighter page doesn't exist)
            if response.url != response.request.url:
                self.logger.warning(f"Redirected from {response.request.url} to {response.url}")
                failed_urls.append((url_fighter_name, "redirected"))

            name = response.css('h1.hero-profile__name::text').get()
            nickname = response.css('p.hero-profile__nickname::text').get()
            division = response.css('p.hero-profile__division-title::text').get()
            record = response.css('p.hero-profile__division-body::text').get()

            if not name or not name.strip():
                self.logger.info(f"Name not found on page: {response}")
                failed_urls.append((url_fighter_name, "no_name_found"))
                return #Skip this item

            # Extract wins, losses, draws from record
            # Example input: '38-19-0 (W-L-D)'
            if record:
                try:
                    parts = record.strip().split(' ')[0]  # Get '38-19-0'
                    wins, losses, draws = parts.split('-')  # Split into separate values
                except (ValueError, IndexError):
                    self.logger.warning(f"Could not parse record '{record}' for {name}")
                    wins = losses = draws = '0'
            else:
                wins = losses = draws = '0'

            # Initialize stats dictionary
            stats = {}

            # Loop through all stat blocks
            for stat in response.css('div.hero-profile__stat'):
                stat_value = stat.css('p.hero-profile__stat-numb::text').get()
                stat_label = stat.css('p.hero-profile__stat-text::text').get()

                if stat_label and stat_value:
                    stat_label = stat_label.strip().lower()

                    if 'wins by knockout' in stat_label:
                        stats['wins_by_knockout'] = stat_value.strip()

                    #if 'wins by submission' in stat_label:
                    #    stats['wins_by_submission'] = stat_value.strip()

                    if 'first round finish' in stat_label:
                        stats['first_round_finishes'] = stat_value.strip()


            for stat in response.css('div.c-bio__field'):
                stat_value = stat.css('div.c-bio__text::text').get()
                stat_label = stat.css('div.c-bio__label::text').get()
                # Loop over BIO
                if stat_label:
                    stat_label = stat_label.strip().lower()

                    if 'status' in stat_label:
                        stats['status'] = stat_value.strip() if stat_value else ''

                    elif 'place of birth' in stat_label:
                        stats['place_of_birth'] = stat_value.strip() if stat_value else ''

                    elif 'trains at' in stat_label:
                        stats['trains_at'] = stat_value.strip() if stat_value else ''

                    elif 'fighting style' in stat_label:
                        stats['fighting_style'] = stat_value.strip() if stat_value else ''

                    elif 'age' in stat_label:
                        # Directly target the inner div that holds the age
                        stat_value = stat.css('div.field--name-age::text').get()
                        stats['age'] = stat_value.strip() if stat_value else ''

                    elif 'height' in stat_label:
                        stats['height'] = stat_value.strip() if stat_value else ''

                    elif 'weight' in stat_label:
                        stats['weight'] = stat_value.strip() if stat_value else ''

                    elif 'octagon debut' in stat_label:
                        stats['octagon_debut'] = stat_value.strip() if stat_value else ''

                    elif 'reach' in stat_label and 'leg' not in stat_label:
                        stats['reach'] = stat_value.strip() if stat_value else ''

                    elif 'leg reach' in stat_label:
                        stats['leg_reach'] = stat_value.strip() if stat_value else ''


            for stat in response.css('dl.c-overlap__stats'):
                stat_value = stat.css('dd.c-overlap__stats-value::text').get()
                stat_label = stat.css('dt.c-overlap__stats-text::text').get()
                # Loop over the first two groups
                if stat_label and stat_value:
                    stat_label = stat_label.strip().lower()

                    if 'sig. strikes landed' in stat_label:
                        stats['sig_strikes_landed'] = stat_value.strip()

                    if 'sig. strikes attempted' in stat_label:
                        stats['sig_strikes_attempted'] = stat_value.strip()

                    if 'takedowns landed' in stat_label:
                        stats['takedowns_landed'] = stat_value.strip()

                    if 'takedowns attempted' in stat_label:
                        stats['takedowns_attempted'] = stat_value.strip()


            for stat in response.css('div.c-stat-compare.c-stat-compare--no-bar'):
                # Loop over the left and right groups
                for group in stat.css('div.c-stat-compare__group'):
                    stat_value = group.css('div.c-stat-compare__number::text').get()
                    stat_label = group.css('div.c-stat-compare__label::text').get()

                    if stat_label and stat_value:
                        stat_label = stat_label.strip().lower()

                        if 'sig. str. landed' in stat_label:
                            stats['sig_strikes_landed_per_minute'] = stat_value.strip()

                        if 'sig. str. absorbed' in stat_label:
                            stats['sig_strikes_absorbed_per_minute'] = stat_value.strip()

                        if 'takedown avg' in stat_label:
                            stats['takedowns_avg_per_15_minute'] = stat_value.strip()

                        if 'submission avg' in stat_label:
                            stats['submission_avg_per_15_minute'] = stat_value.strip()

                        if 'sig. str. defense' in stat_label:
                            stats['sig_strikes_defense_%'] = stat_value.strip()

                        if 'takedown defense' in stat_label:
                            stats['takedown_defense_%'] = stat_value.strip()

                        if 'knockdown avg' in stat_label:
                            stats['knockdown_avg'] = stat_value.strip()

                        if 'average fight time' in stat_label:
                            stats['fight_time_avg'] = stat_value.strip()


            for stat in response.css('div.c-stat-3bar__legend'):
                # Loop over the left and right groups
                for group in stat.css('div.c-stat-3bar__group'):
                    stat_value = group.css('div.c-stat-3bar__value::text').get()
                    stat_label = group.css('div.c-stat-3bar__label::text').get()

                    if stat_label and stat_value:
                        stat_label = stat_label.strip().lower()
                        stat_value = stat_value.strip().split(' ')[0]

                        if 'standing' in stat_label:
                            stats['sig_strikes_standing'] = stat_value.strip()

                        if 'clinch' in stat_label:
                            stats['sig_strikes_clinch'] = stat_value.strip()

                        if 'ground' in stat_label:
                            stats['sig_strikes_ground'] = stat_value.strip()

                        if 'ko/tko' in stat_label:
                            stats['win_by_ko/tko'] = stat_value.strip()

                        if 'dec' in stat_label:
                            stats['win_by_dec'] = stat_value.strip()

                        if 'sub' in stat_label:
                            stats['win_by_sub'] = stat_value.strip()
                            

            target_areas = ['head', 'body', 'leg']
            # Loop over sig str by target group
            for area in target_areas:
                group = response.css(f'g#e-stat-body_x5F__x5F_{area}-txt')

                if group:
                    value = group.css(f'text#e-stat-body_x5F__x5F_{area}_value::text').get()
                    label = group.css('text::text')[-1].get()  # The last text element is the label like 'Head'

                    if label and value:
                        stats[f'{area}_target'] = value.strip()


            yield { 
                #'fighter_info'

                'name': name.strip() if name else None,
                'nickname': nickname.strip() if nickname else None,
                'division': division.strip() if division else None,
                'record': record.strip() if record else None,
                'status': stats.get('status', None),
                'place_of_birth': stats.get('place_of_birth', None),
                'trains_at': stats.get('trains_at', None),
                'fighting_style': stats.get('fighting_style', None),
                'octagon_debut': stats.get('octagon_debut', None),
                'age': stats.get('age', None),
                'height': stats.get('height', None),
                'weight': stats.get('weight', None),
                'reach': stats.get('reach', None),
                'leg_reach': stats.get('leg_reach', None),

                #'fighter_stats'

                'wins': wins,
                'losses': losses,
                'draws': draws,
                'wins_by_knockout': stats.get('wins_by_knockout', None),
                #'wins_by_submission': stats.get('wins_by_submission', '0'),
                'first_round_finishes': stats.get('first_round_finishes', None),
                'win_by_dec': stats.get('win_by_dec', None),
                'win_by_sub': stats.get('win_by_sub', None),
                'sig_strikes_landed': stats.get('sig_strikes_landed', None),
                'sig_strikes_attempted': stats.get('sig_strikes_attempted', None),
                'takedowns_landed': stats.get('takedowns_landed', None),
                'takedowns_attempted': stats.get('takedowns_attempted', None),
                'sig_strikes_landed_per_minute': stats.get('sig_strikes_landed_per_minute', None),
                'sig_strikes_absorbed_per_minute': stats.get('sig_strikes_absorbed_per_minute', None),
                'takedowns_avg': stats.get('takedowns_avg_per_15_minute', None),
                'submission_avg': stats.get('submission_avg_per_15_minute', None),
                'sig_strikes_defense': stats.get('sig_strikes_defense_%', None),
                'takedown_defense': stats.get('takedown_defense_%', None),
                'knockdown_avg': stats.get('knockdown_avg', None),
                'fight_time_avg': stats.get('fight_time_avg', None),
                'sig_strikes_standing': stats.get('sig_strikes_standing', None),
                'sig_strikes_clinch': stats.get('sig_strikes_clinch', None),
                'sig_strikes_ground': stats.get('sig_strikes_ground', None),
                'head_target': stats.get('head_target', None),
                'body_target': stats.get('body_target', None),
                'leg_target': stats.get('leg_target', None),     
                
            }

    def errback_httpbin(self, failure):
        # Handle request failures
        request = failure.request
        url_fighter_name = request.url.split('/')[-1]
        self.logger.error(f"Request failed for {url_fighter_name}: {failure}")
        failed_urls.append((url_fighter_name, str(failure.value)))

# Use csv
'''
# Merge updated data into existing master file
def update_fighter_stats(old_file, updated_file, key_columns, update_columns):
    if not os.path.exists(updated_file):
        print("Update filenot found")
        return
    
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(updated_file)

    merged_df = pd.merge(old_df, new_df[key_columns + update_columns],
                        on=key_columns,
                        how='left',
                        suffixes=('','_new'))
    
    # Replace only the updated columns
    for col in update_columns:
        new_col = col + '_new'
        if new_col in merged_df.columns:
            merged_df[col] = merged_df[new_col].combine_first(merged_df[col])
            merged_df.drop(columns=[new_col], inplace=True)

    merged_df.to_csv(old_file, index=False)
    print(f"Updated master file saved to {old_file}")
'''

def update_fighter_stats_sql(host, user, password, database, table_name, updated_file, key_columns, update_columns):
    
    # Update existing fighter rows in a MySQL database using mysql.connector.
    if not os.path.exists(updated_file):
        print("Updated CSV file not found.")
        return

    new_df = pd.read_csv(updated_file)

    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        cursor = conn.cursor()

        for _, row in new_df.iterrows():
            # Build the SET and WHERE clause
            set_clause = ", ".join([f"{col} = %s" for col in update_columns])
            where_clause = " AND ".join([f"{col} = %s" for col in key_columns])

            sql = f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE {where_clause}
            """

            # Combine values for SET and WHERE in order
            values = [row[col] if pd.notna(row[col]) else None for col in update_columns + key_columns]

            #print("SQL:", sql)
            print("Values:", values)
            cursor.execute(sql, values)
            print(f"Affected rows: {cursor.rowcount}")


        conn.commit()

    except Error as e:
        print(f"MySQL error: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    # Step 1: Get fighter list
    fighter_df = extract_fighters()
    
    # Step 2: Convert to UFC-style URLs
    fighters_csv = fighter_df.applymap(lambda x: x.lower().replace(' ', '-'))
    athlete_names = fighters_csv['fighter'].to_list()

    # Step 3: Scrape data
    output_file = "data/updated_stats.csv"
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
        "RETRY_HTTP_CODES": [403, 429, 500, 502, 503, 504, 522, 524],
        "DOWNLOAD_TIMEOUT": 15,
        "DOWNLOAD_DELAY": 1,
        "RANDOMIZE_DOWNLOAD_DELAY": 0.5,
        "CONCURRENT_REQUESTS": 8,
    })

    process.crawl(UfcAthleteSpider, athlete_names=athlete_names)
    process.start()

    print("Data scraped and saved!")
    print(f"Total fighters attempted: {len(athlete_names)}")
    print(f"Successful scraped: {len(successful_names)}")
    print(f"Failed URLs: {len(failed_urls)}")

    end_time = time.time()
    print(f"\nScraping completed in {end_time - start_time:.2f} seconds.")

    df = pd.read_csv(output_file)
    print(df)

    # Step 4: Merge into master
    #old_master_file = 'data/stats.csv'
    
    # Test database connection first
    print("Testing database connection...")
    try:
        database=os.getenv("DB_NAME")
        conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        database=os.getenv("DB_NAME"),
        password=os.getenv("DB_PASSWORD")
        )

        cursor = conn.cursor()
        print(f"✓ Successfully connected to database {database}")
            
    except Error as e:
        print(f"✗ Database connection failed: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    # Initliaze columns
    key_columns = ['name', 'place_of_birth'] 
    update_columns = [
        'nickname', 'octagon_debut',
        'division','record','status','trains_at','fighting_style','age','height','weight','reach','leg_reach',
        'wins','losses','draws','wins_by_knockout','first_round_finishes','win_by_dec','win_by_sub',
        'sig_strikes_landed','sig_strikes_attempted','takedowns_landed','takedowns_attempted',
        'sig_strikes_landed_per_minute','sig_strikes_absorbed_per_minute','takedowns_avg','submission_avg',
        'sig_strikes_defense','takedown_defense','knockdown_avg','fight_time_avg',
        'sig_strikes_standing','sig_strikes_clinch','sig_strikes_ground',
        'head_target','body_target','leg_target'
    ]

    #Update csv
    #update_fighter_stats(old_master_file, output_file, key_columns, update_columns)

    # Step 5: Update MySQL database
    update_fighter_stats_sql(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    database=os.getenv("DB_NAME"),
    password=os.getenv("DB_PASSWORD"),
    table_name="stats",
    updated_file=output_file,
    key_columns=key_columns,
    update_columns=update_columns)

    # Step 6: Check for missing/new fighters
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        database=os.getenv("DB_NAME"),
        password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM stats")  
    db_fighters = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    scraped_fighters = df["name"].dropna().tolist()
    missing_fighters = [f for f in db_fighters if f not in scraped_fighters]
    new_fighters = [f for f in scraped_fighters if f not in db_fighters]

    if missing_fighters:
        pd.DataFrame({"missing_fighter": missing_fighters}).to_csv("data/missing_fighters.csv", index=False)
        print(f"Missing fighters saved to data/missing_fighters.csv")

    # Remove temporarily file
    #os.remove(output_file)

    print("Fighters to update:")
    print(new_fighters)
    print("\nTotal New Fighters Found: ", len(new_fighters))

    print("\nFighters not found:")
    #print(missing_fighters)
    print("\nTotal Fighters Not Found: ", len(missing_fighters))

# Constant variables per fighter
#name, place_of_birth, 
#nickname, octagon_debut

# Changing variables per fighter
#division,record,status,trains_at,fighting_style,age,height,weight,reach,leg_reach,wins,losses,draws,wins_by_knockout,first_round_finishes,win_by_dec,win_by_sub,sig_strikes_landed,sig_strikes_attempted,takedowns_landed,takedowns_attempted,sig_strikes_landed_per_minute,sig_strikes_absorbed_per_minute,takedowns_avg,submission_avg,sig_strikes_defense,takedown_defense,knockdown_avg,fight_time_avg,sig_strikes_standing,sig_strikes_clinch,sig_strikes_ground,head_target,body_target,leg_target

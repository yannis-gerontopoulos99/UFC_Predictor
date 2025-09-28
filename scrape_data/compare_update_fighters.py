import pandas as pd
import os
import scrapy
from scrapy.crawler import CrawlerProcess
import time
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import unicodedata
import re
from fuzzywuzzy import fuzz
from contextlib import contextmanager

load_dotenv()

class DatabaseManager:
    """Handle all database operations"""
    
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.user = os.getenv("DB_USER") 
        self.database = os.getenv("DB_NAME")
        self.password = os.getenv("DB_PASSWORD")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                database=self.database,
                password=self.password
            )
            cursor = conn.cursor()
            yield conn, cursor
        except Error as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def test_connection(self):
        """Test database connection"""
        print("Testing database connection...")
        try:
            with self.get_connection() as (conn, cursor):
                print(f"✓ Successfully connected to database {self.database}")
                return True
        except Error as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def extract_fighters_from_db(self):
        """Extract all fighters from database"""
        with self.get_connection() as (conn, cursor):
            cursor.execute("SELECT fighter FROM fighters")
            return [row[0] for row in cursor.fetchall()]
    
    def insert_new_fighters(self, fighters_df):
        """Insert new fighters into database"""
        if fighters_df is None or fighters_df.empty:
            print("No fighters to insert")
            return 0, []
            
        success_count = 0
        failed_rows = []
        
        with self.get_connection() as (conn, cursor):
            insert_query = "INSERT INTO fighters (fighter) VALUES (%s)"
            
            for i, row in fighters_df[["original_fighter"]].iterrows():
                try:
                    cursor.execute(insert_query, tuple(row))
                    success_count += 1
                except Exception as e:
                    failed_rows.append((i, row.to_dict(), str(e)))
            
            conn.commit()
        
        print(f"New fighters added: {success_count}")
        print(f"Failed entries: {len(failed_rows)}")
        return success_count, failed_rows

class FighterDataProcessor:
    """Handle fighter data processing and matching"""
    
    @staticmethod
    def extract_fighters_from_bouts(temp_file='data/temp_bouts.csv'):
        """Extract fighters and convert into one column"""
        if not os.path.exists(temp_file):
            print(f"File not found: {temp_file}")
            return pd.DataFrame()
        
        output_df = pd.read_csv(temp_file)
        
        if 'fighter_blue' not in output_df.columns or 'fighter_red' not in output_df.columns:
            print("Columns 'fighter_blue' and/or 'fighter_red not found in the file")
            return pd.DataFrame()
        
        # Concat two columns together
        fighter_series = pd.concat([output_df['fighter_blue'], output_df['fighter_red']], ignore_index=True)
        fighter_series = fighter_series.dropna().astype(str)
        
        return pd.DataFrame({'fighter': fighter_series})
    
    @staticmethod
    def normalize_name(name):
        """Normalize fighter names for comparison"""
        if pd.isna(name):
            return ""
        
        # Convert to string if not already
        name = str(name)
        # Normalize unicode characters
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        # Convert to lower
        name = name.lower()
        # Remove periods and other punctuation
        name = re.sub(r'[^\w\s-]', '', name)
        # Normalize spaces (multiple spaces to single space)
        name = re.sub(r'\s+', ' ', name)
        words = name.split()
        
        return ' '.join(words).strip()
    
    def create_fuzzy_mapping(self, output_names, db_names, threshold=85):
        """Map event fighter names to the closest match in db_names"""
        mapping = {}
        exact_matches = 0
        fuzzy_matches = 0
        no_matches = 0
        no_matches_list = []
        
        for output_name in output_names:
            if pd.isna(output_name):
                continue
                
            best_match = None
            best_score = 0
            
            for db_name in db_names:
                if pd.isna(db_name):
                    continue
                
                # Compare event name with db name
                score = max(
                    fuzz.ratio(output_name, db_name),
                    fuzz.token_sort_ratio(output_name, db_name),
                    fuzz.token_set_ratio(output_name, db_name)
                )
                
                if score > best_score:
                    best_match = db_name
                    best_score = score
            
            # Decide outcome
            if best_score >= threshold:
                mapping[output_name] = best_match
                if output_name == best_match:
                    exact_matches += 1
                else:
                    fuzzy_matches += 1
            else:
                mapping[output_name] = None
                no_matches += 1
                no_matches_list.append(output_name)
        
        # Convert no matches into a DataFrame (if any)
        no_matches_df = pd.DataFrame(no_matches_list, columns=["unmatched_fighter"]) if no_matches_list else None
        
        stats = {
            "total_output_fighters": len(set(output_names)),
            "total_db_names": len(set(db_names)),
            "exact_matches": exact_matches,
            "fuzzy_matches": fuzzy_matches,
            "no_matches": no_matches
        }
        
        self._print_mapping_stats(stats, no_matches_df)
        
        return mapping, stats, no_matches_df
    
    def _print_mapping_stats(self, stats, no_matches_df):
        """Print mapping statistics"""
        print(f"Total unique fighters in events: {stats['total_output_fighters']}")
        print(f"Total unique names in stats: {stats['total_db_names']}")
        print(f"Exact matches: {stats['exact_matches']}")
        print(f"Fuzzy matches: {stats['fuzzy_matches']}")
        print(f"No matches found: {stats['no_matches']}")
        
        if stats['no_matches'] > 0 and no_matches_df is not None:
            print("\nUnmatched fighters:")
            print(no_matches_df)
    
    def prepare_unmatched_fighters(self, no_matches_df, fighter_output_df):
        """Map normalized unmatched names back to original names"""
        if no_matches_df is None:
            return None
            
        # Map back normalized → original
        norm_to_orig = dict(zip(fighter_output_df["normalized_fighter"], fighter_output_df["fighter"]))
        no_matches_df["original_fighter"] = no_matches_df["unmatched_fighter"].map(norm_to_orig)
        
        print("\nUnmatched fighters in original format:")
        print(no_matches_df[["original_fighter"]])
        
        return no_matches_df

class UFCScrapingManager:
    """Manage UFC athlete scraping operations"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def prepare_fighter_names_for_scraping(self, no_matches_df):
        """Prepare fighter names in URL-friendly format"""
        if no_matches_df is None or no_matches_df.empty:
            return []
            
        fighters_csv = no_matches_df[["original_fighter"]].copy()
        fighters_csv = fighters_csv.applymap(lambda x: x.lower().replace(' ', '-'))
        return fighters_csv['original_fighter'].to_list()
    
    def create_scraper_process(self, athlete_names):
        """Create and configure scrapy process"""
        process = CrawlerProcess(settings={
            "FEEDS": {
                "output.csv": {
                    "format": "csv",
                    "encoding": "utf8",
                    "overwrite": True,
                    "store_empty": True
                },
            },
            "USER_AGENT": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/115.0.0.0 Safari/537.36',
            "ROBOTSTXT_OBEY": False,
            "LOG_LEVEL": "INFO",
            "RETRY_ENABLED": True,
            "RETRY_HTTP_CODES": [403, 500, 502, 503, 504],
            "DOWNLOAD_TIMEOUT": 15,
            "ITEM_PIPELINES": {
                "__main__.MySQLStorePipeline": 1,
            },
            "LOG_ENABLED": True,
        })
        
        return process
    
    def run_scraping(self, athlete_names):
        """Execute the scraping process"""
        if not athlete_names:
            print("No athlete names to scrape")
            return
            
        start_time = time.time()
        
        process = self.create_scraper_process(athlete_names)
        process.crawl(UfcAthleteSpider, athlete_names=athlete_names)
        process.start()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Scraping completed in {elapsed_time:.2f} seconds.")

class UfcAthleteSpider(scrapy.Spider):
    """Scrapy spider for UFC athlete data"""
    name = "ufc_athlete"
    allowed_domains = ["ufc.com"]
    
    def __init__(self, athlete_names=None, *args, **kwargs):
        super(UfcAthleteSpider, self).__init__(*args, **kwargs)
        self.start_urls = [f"https://www.ufc.com/athlete/{name}" for name in (athlete_names or [])]
    
    def parse(self, response):
        """Parse UFC athlete page"""
        # Extract basic info
        name = response.css('h1.hero-profile__name::text').get()
        if not name or not name.strip():
            self.logger.info(f"Name not found on page: {response}")
            return
        
        nickname = response.css('p.hero-profile__nickname::text').get()
        division = response.css('p.hero-profile__division-title::text').get()
        record = response.css('p.hero-profile__division-body::text').get()
        
        # Parse record
        wins, losses, draws = self._parse_record(record)
        
        # Extract all stats
        stats = {}
        self._extract_hero_stats(response, stats)
        self._extract_bio_stats(response, stats)
        self._extract_overlap_stats(response, stats)
        self._extract_compare_stats(response, stats)
        self._extract_bar_stats(response, stats)
        self._extract_target_stats(response, stats)
        
        yield self._build_item(name, nickname, division, record, wins, losses, draws, stats)
    
    def _parse_record(self, record):
        """Parse fight record string"""
        if record:
            parts = record.strip().split(' ')[0]  # Get '38-19-0'
            wins, losses, draws = parts.split('-')
        else:
            wins = losses = draws = '0'
        return wins, losses, draws
    
    def _extract_hero_stats(self, response, stats):
        """Extract hero profile stats"""
        for stat in response.css('div.hero-profile__stat'):
            stat_value = stat.css('p.hero-profile__stat-numb::text').get()
            stat_label = stat.css('p.hero-profile__stat-text::text').get()
            
            if stat_label and stat_value:
                stat_label = stat_label.strip().lower()
                
                if 'wins by knockout' in stat_label:
                    stats['wins_by_knockout'] = stat_value.strip()
                elif 'first round finish' in stat_label:
                    stats['first_round_finishes'] = stat_value.strip()
    
    def _extract_bio_stats(self, response, stats):
        """Extract bio field stats"""
        bio_mapping = {
            'status': 'status',
            'place of birth': 'place_of_birth',
            'trains at': 'trains_at',
            'fighting style': 'fighting_style',
            'age': 'age',
            'height': 'height',
            'weight': 'weight',
            'octagon debut': 'octagon_debut',
            'reach': 'reach',
            'leg reach': 'leg_reach'
        }
        
        for stat in response.css('div.c-bio__field'):
            stat_value = stat.css('div.c-bio__text::text').get()
            stat_label = stat.css('div.c-bio__label::text').get()
            
            if stat_label:
                stat_label = stat_label.strip().lower()
                
                for key, field in bio_mapping.items():
                    if key in stat_label:
                        if key == 'age':
                            stat_value = stat.css('div.field--name-age::text').get()
                        stats[field] = stat_value.strip() if stat_value else ''
                        break
    
    def _extract_overlap_stats(self, response, stats):
        """Extract overlap stats"""
        overlap_mapping = {
            'sig. strikes landed': 'sig_strikes_landed',
            'sig. strikes attempted': 'sig_strikes_attempted',
            'takedowns landed': 'takedowns_landed',
            'takedowns attempted': 'takedowns_attempted'
        }
        
        for stat in response.css('dl.c-overlap__stats'):
            stat_value = stat.css('dd.c-overlap__stats-value::text').get()
            stat_label = stat.css('dt.c-overlap__stats-text::text').get()
            
            if stat_label and stat_value:
                stat_label = stat_label.strip().lower()
                
                for key, field in overlap_mapping.items():
                    if key in stat_label:
                        stats[field] = stat_value.strip()
                        break
    
    def _extract_compare_stats(self, response, stats):
        """Extract comparison stats"""
        compare_mapping = {
            'sig. str. landed': 'sig_strikes_landed_per_minute',
            'sig. str. absorbed': 'sig_strikes_absorbed_per_minute',
            'takedown avg': 'takedowns_avg_per_15_minute',
            'submission avg': 'submission_avg_per_15_minute',
            'sig. str. defense': 'sig_strikes_defense_%',
            'takedown defense': 'takedown_defense_%',
            'knockdown avg': 'knockdown_avg',
            'average fight time': 'fight_time_avg'
        }
        
        for stat in response.css('div.c-stat-compare.c-stat-compare--no-bar'):
            for group in stat.css('div.c-stat-compare__group'):
                stat_value = group.css('div.c-stat-compare__number::text').get()
                stat_label = group.css('div.c-stat-compare__label::text').get()
                
                if stat_label and stat_value:
                    stat_label = stat_label.strip().lower()
                    
                    for key, field in compare_mapping.items():
                        if key in stat_label:
                            stats[field] = stat_value.strip()
                            break
    
    def _extract_bar_stats(self, response, stats):
        """Extract 3-bar stats"""
        bar_mapping = {
            'standing': 'sig_strikes_standing',
            'clinch': 'sig_strikes_clinch',
            'ground': 'sig_strikes_ground',
            'ko/tko': 'win_by_ko/tko',
            'dec': 'win_by_dec',
            'sub': 'win_by_sub'
        }
        
        for stat in response.css('div.c-stat-3bar__legend'):
            for group in stat.css('div.c-stat-3bar__group'):
                stat_value = group.css('div.c-stat-3bar__value::text').get()
                stat_label = group.css('div.c-stat-3bar__label::text').get()
                
                if stat_label and stat_value:
                    stat_label = stat_label.strip().lower()
                    stat_value = stat_value.strip().split(' ')[0]
                    
                    for key, field in bar_mapping.items():
                        if key in stat_label:
                            stats[field] = stat_value.strip()
                            break
    
    def _extract_target_stats(self, response, stats):
        """Extract target area stats"""
        target_areas = ['head', 'body', 'leg']
        
        for area in target_areas:
            group = response.css(f'g#e-stat-body_x5F__x5F_{area}-txt')
            
            if group:
                value = group.css(f'text#e-stat-body_x5F__x5F_{area}_value::text').get()
                if value:
                    stats[f'{area}_target'] = value.strip()
    
    def _build_item(self, name, nickname, division, record, wins, losses, draws, stats):
        """Build the final item dictionary"""
        return {
            # Fighter info
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
            
            # Fighter stats
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'wins_by_knockout': stats.get('wins_by_knockout', None),
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

class MySQLStorePipeline:
    """Pipeline to store scraped items in MySQL"""
    
    def open_spider(self, spider):
        self.db_manager = DatabaseManager()
        self.conn, self.cursor = self.db_manager.get_connection().__enter__()
        self.success_count = 0
        self.failed_rows = []
    
    def close_spider(self, spider):
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        
        spider.logger.info(f"Successfully inserted: {self.success_count}")
        spider.logger.info(f"Failed inserts: {len(self.failed_rows)}")
    
    def process_item(self, item, spider):
        table = 'stats'
        columns = ', '.join(item.keys())
        placeholders = ', '.join(['%s'] * len(item))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        try:
            self.cursor.execute(sql, list(item.values()))
            self.success_count += 1
        except mysql.connector.Error as err:
            self.failed_rows.append({**item, "error": str(err)})
            spider.logger.error(f"Failed to insert item: {err}")
        
        return item

def process_fighters_and_get_new_names(fighter_output_df):
    """
    Process fighters against database and return new fighter names for scraping.
    This is the main function to be called from other scripts.
    
    Args:
        fighter_output_df: DataFrame with 'fighter' column containing fighter names
        
    Returns:
        tuple: (athlete_names_list, stats_dict) where:
            - athlete_names_list: List of URL-formatted names for scraping
            - stats_dict: Dictionary with matching statistics
    """
    # Initialize managers
    db_manager = DatabaseManager()
    processor = FighterDataProcessor()
    scraping_manager = UFCScrapingManager(db_manager)
    
    if fighter_output_df.empty:
        print("No fighters provided.")
        return [], {}
    
    # Test database connection
    print("Testing database connection...")
    if not db_manager.test_connection():
        print("Database connection failed.")
        return [], {}
    
    # Extract fighters from database
    print("Extracting fighters from database...")
    fighter_database_list = db_manager.extract_fighters_from_db()
    
    # Normalize names
    print("Normalizing fighter names...")
    fighter_output_df["normalized_fighter"] = fighter_output_df["fighter"].apply(processor.normalize_name)
    normalized_fighters_db = [processor.normalize_name(name) for name in fighter_database_list]
    
    # Run fuzzy mapping
    print("Running fuzzy mapping...")
    mapping, stats, no_matches_df = processor.create_fuzzy_mapping(
        fighter_output_df["normalized_fighter"], 
        normalized_fighters_db
    )
    
    # Handle unmatched fighters
    if no_matches_df is not None and not no_matches_df.empty:
        print("Processing unmatched fighters...")
        
        # Prepare unmatched fighters
        no_matches_df = processor.prepare_unmatched_fighters(no_matches_df, fighter_output_df)
        
        # Insert new fighters into database
        success_count, failed_rows = db_manager.insert_new_fighters(no_matches_df)
        
        # Prepare names for scraping (URL format)
        athlete_names = scraping_manager.prepare_fighter_names_for_scraping(no_matches_df)
        
        print(f"Found {len(athlete_names)} new fighters to scrape")
        return athlete_names, stats
    else:
        print("No unmatched fighters found.")
        return [], stats

def run_fighter_scraping(athlete_names):
    """
    Run the scraping process for given athlete names.
    
    Args:
        athlete_names: List of URL-formatted fighter names
    """
    if not athlete_names:
        print("No athlete names provided for scraping.")
        return
        
    db_manager = DatabaseManager()
    scraping_manager = UFCScrapingManager(db_manager)
    
    print(f"Starting scraping for {len(athlete_names)} fighters...")
    scraping_manager.run_scraping(athlete_names)

def process_fighters_and_get_new_names(fighter_output_df):
    """
    Process fighters against database and return new fighter names for scraping.
    This is the main function to be called from other scripts.
    
    Args:
        fighter_output_df: DataFrame with 'fighter' column containing fighter names
        
    Returns:
        tuple: (athlete_names_list, stats_dict) where:
            - athlete_names_list: List of URL-formatted names for scraping
            - stats_dict: Dictionary with matching statistics
    """
    # Initialize managers
    db_manager = DatabaseManager()
    processor = FighterDataProcessor()
    scraping_manager = UFCScrapingManager(db_manager)
    
    if fighter_output_df.empty:
        print("No fighters provided.")
        return [], {}
    
    # Test database connection
    print("Testing database connection...")
    if not db_manager.test_connection():
        print("Database connection failed.")
        return [], {}
    
    # Extract fighters from database
    print("Extracting fighters from database...")
    fighter_database_list = db_manager.extract_fighters_from_db()
    
    # Normalize names
    print("Normalizing fighter names...")
    fighter_output_df["normalized_fighter"] = fighter_output_df["fighter"].apply(processor.normalize_name)
    normalized_fighters_db = [processor.normalize_name(name) for name in fighter_database_list]
    
    # Run fuzzy mapping
    print("Running fuzzy mapping...")
    mapping, stats, no_matches_df = processor.create_fuzzy_mapping(
        fighter_output_df["normalized_fighter"], 
        normalized_fighters_db
    )
    
    # Handle unmatched fighters
    if no_matches_df is not None and not no_matches_df.empty:
        print("Processing unmatched fighters...")
        
        # Prepare unmatched fighters
        no_matches_df = processor.prepare_unmatched_fighters(no_matches_df, fighter_output_df)
        
        # Insert new fighters into database
        success_count, failed_rows = db_manager.insert_new_fighters(no_matches_df)
        
        # Prepare names for scraping (URL format)
        athlete_names = scraping_manager.prepare_fighter_names_for_scraping(no_matches_df)
        
        print(f"Found {len(athlete_names)} new fighters to scrape")
        return athlete_names, stats
    else:
        print("No unmatched fighters found.")
        return [], stats

def main():
    """Main execution function for standalone use"""
    # Initialize processor
    processor = FighterDataProcessor()
    
    # Extract fighters from bouts
    print("Step 1: Extracting fighters from bouts...")
    fighter_output_df = processor.extract_fighters_from_bouts()
    
    if fighter_output_df.empty:
        print("No fighters found. Exiting.")
        return
    
    # Process fighters and get new names
    athlete_names, stats = process_fighters_and_get_new_names(fighter_output_df)
    
    # Run scraping if there are new fighters
    if athlete_names:
        run_fighter_scraping(athlete_names)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
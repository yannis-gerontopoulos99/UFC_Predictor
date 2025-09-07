import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
import time
import os

# Read all fighter names
fighters_csv = pd.read_csv('data/fighters.csv')
# Convert to the url format to read the names
fighters_csv = fighters_csv.applymap(lambda x: x.lower().replace(' ', '-'))
athlete_names = fighters_csv['fighter'].to_list()

# Track failed URLs and reasons
failed_urls = []
successful_names = []

# Scraping method
class UfcAthleteSpider(scrapy.Spider):
    name = "ufc_athlete"
    # Domain name
    allowed_domains = ["ufc.com"]
    # Url to start scraping from using the fighter names
    start_urls = [f"https://www.ufc.com/athlete/{name}" for name in athlete_names]

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

if __name__ == "__main__":

    start_time = time.time()

    output_file = "data/stats1.csv"
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

    # Start process
    process.crawl(UfcAthleteSpider) 
    process.start()

    print("Data scraped and saved!")
    print(f"Total fighters attempted: {len(athlete_names)}")
    print(f"Successful scraped: {len(successful_names)}")
    print(f"Failed URLs: {len(failed_urls)}")

    # Calculate duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Scraping completed in {elapsed_time:.2f} seconds.")

    print(f"Data saved and sorted to: {os.path.abspath(output_file)}")


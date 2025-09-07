import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv()  # load variables from .env file

# CONFIGURATION
csv_file_path = 'data/events.csv'
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = 3306
db_name = os.getenv("DB_NAME")
table_name = 'events'

db_password_enc = urllib.parse.quote_plus(db_password)

# READ CSV
df = pd.read_csv(csv_file_path)

# CREATE DB CONNECTION
engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")


# COLUMN TYPE MAPPING
column_types = {
    "event_date": "DATE",
    "event_name": "VARCHAR(255)",
    "fighter_red": "VARCHAR(100)",
    "fighter_blue": "VARCHAR(100)",
    "round": "SMALLINT UNSIGNED", 
    "time": "SMALLINT UNSIGNED",
    "weight_class": "VARCHAR(100)",
    "win_method": "VARCHAR(100)",
    "winner": "VARCHAR(100)",
    "stance_red": "VARCHAR(100)",
    "stance_blue": "VARCHAR(100)",
    "knockdowns_red": "SMALLINT UNSIGNED",
    "knockdowns_blue": "SMALLINT UNSIGNED",
    "sig_attempts_red": "SMALLINT UNSIGNED",
    "sig_attempts_blue": "SMALLINT UNSIGNED",
    "sig_strikes_red": "SMALLINT UNSIGNED",
    "sig_strikes_blue": "SMALLINT UNSIGNED",
    "total_strikes_attempts_red": "SMALLINT UNSIGNED",
    "total_strikes_attempts_blue": "SMALLINT UNSIGNED",
    "total_strikes_red": "SMALLINT UNSIGNED",
    "total_strikes_blue": "SMALLINT UNSIGNED", 
    "sub_attempts_red": "SMALLINT UNSIGNED",
    "sub_attempts_blue": "SMALLINT UNSIGNED",
    "takedowns_red": "SMALLINT UNSIGNED",
    "takedowns_blue": "SMALLINT UNSIGNED",
    "takedown_attempts_red": "SMALLINT UNSIGNED",
    "takedown_attempts_blue": "SMALLINT UNSIGNED",
    "control_time_red": "SMALLINT UNSIGNED",
    "control_time_blue": "SMALLINT UNSIGNED",
    "head_strikes_red": "SMALLINT UNSIGNED",
    "head_strikes_blue": "SMALLINT UNSIGNED",
    "head_attempts_red": "SMALLINT UNSIGNED",
    "head_attempts_blue": "SMALLINT UNSIGNED",
    "body_strikes_red": "SMALLINT UNSIGNED",
    "body_strikes_blue": "SMALLINT UNSIGNED",
    "body_attempts_red": "SMALLINT UNSIGNED",
    "body_attempts_blue": "SMALLINT UNSIGNED",
    "leg_strikes_red": "SMALLINT UNSIGNED",
    "leg_strikes_blue": "SMALLINT UNSIGNED",
    "leg_attempts_red": "SMALLINT UNSIGNED",
    "leg_attempts_blue": "SMALLINT UNSIGNED",
    "distance_red": "SMALLINT UNSIGNED",
    "distance_blue": "SMALLINT UNSIGNED",
    "distance_attempts_red": "SMALLINT UNSIGNED",
    "distance_attempts_blue": "SMALLINT UNSIGNED",
    "clinch_strikes_red": "SMALLINT UNSIGNED",
    "clinch_strikes_blue": "SMALLINT UNSIGNED",
    "clinch_attempts_red": "SMALLINT UNSIGNED",
    "clinch_attempts_blue": "SMALLINT UNSIGNED",
    "ground_strikes_red": "SMALLINT UNSIGNED",
    "ground_strikes_blue": "SMALLINT UNSIGNED",
    "ground_attempts_red": "SMALLINT UNSIGNED",
    "ground_attempts_blue": "SMALLINT UNSIGNED"
}

# CREATE TABLE
with engine.connect() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

    column_defs = ",\n    ".join([f"`{col}` {dtype}" for col, dtype in column_types.items()])

    create_sql = f"""
    CREATE TABLE `{table_name}` (
        `id` INT AUTO_INCREMENT PRIMARY KEY,
        {column_defs}
    )
    """
    conn.execute(text(create_sql))

# APPEND DATA
df.to_sql(table_name, con=engine, if_exists='append', index=False)

print("Data saved to DB")
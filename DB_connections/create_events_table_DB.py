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
    "fighter_blue": "VARCHAR(100)",
    "fighter_red": "VARCHAR(100)",
    "knockdowns_blue": "INT",
    "knockdowns_red": "INT",
    "sig_attempts_blue": "INT",
    "sig_attempts_red": "INT",
    "sig_strikes_blue": "INT",
    "sig_strikes_red": "INT",
    "total_strikes_attempts_blue": "INT",
    "total_strikes_attempts_red": "INT",
    "total_strikes_blue": "INT",
    "total_strikes_red": "INT", 
    "sub_attempts_blue": "INT",
    "sub_attempts_red": "INT",
    "takedowns_blue": "INT",
    "takedowns_red": "INT",
    "takedown_attempts_blue": "INT",
    "takedown_attempts_red": "INT",
    "control_time_blue": "INT",
    "control_time_red": "INT",
    "round": "INT", 
    "time": "INT",
    "weight_class": "VARCHAR(100)",
    "win_method": "VARCHAR(100)",
    "winner": "VARCHAR(100)"
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
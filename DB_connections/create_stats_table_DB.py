import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv()  # load variables from .env file

# CONFIGURATION
csv_file_path = 'data/stats.csv'
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = 3306
db_name = os.getenv("DB_NAME")
table_name = 'stats'

db_password_enc = urllib.parse.quote_plus(db_password)

# READ CSV
df = pd.read_csv(csv_file_path)

# CREATE DB CONNECTION
engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")


# COLUMN TYPE MAPPING
column_types = {
    "name": "VARCHAR(100)",
    "nickname": "VARCHAR(100)",
    "division": "VARCHAR(100)",
    "record": "VARCHAR(100)",
    "status": "VARCHAR(100)",
    "place_of_birth": "VARCHAR(100)",
    "trains_at": "VARCHAR(100)",
    "fighting_style": "VARCHAR(100)",
    "octagon_debut": "VARCHAR(100)",
    "age": "INT",
    "height": "FLOAT",
    "weight": "FLOAT",
    "reach": "FLOAT",
    "leg_reach": "FLOAT", 
    "wins": "INT",
    "losses": "INT",
    "draws": "INT",
    "wins_by_knockout": "INT",
    #"wins_by_submission": "INT",
    "first_round_finishes": "INT",
    "win_by_dec": "INT",
    "win_by_sub": "INT",
    "sig_strikes_landed": "INT",
    "sig_strikes_attempted": "INT",
    "takedowns_landed": "INT",
    "takedowns_attempted": "INT",
    "sig_strikes_landed_per_minute": "INT",
    "sig_strikes_absorbed_per_minute": "INT",
    "takedowns_avg": "FLOAT",
    "submission_avg": "FLOAT",
    "sig_strikes_defense": "INT",
    "takedown_defense": "INT", 
    "knockdown_avg": "FLOAT",
    "fight_time_avg": "VARCHAR(100)",
    "sig_strikes_standing": "INT",
    "sig_strikes_clinch": "INT",
    "sig_strikes_ground": "INT",
    "head_target": "INT",
    "body_target": "INT",
    "leg_target": "INT"
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
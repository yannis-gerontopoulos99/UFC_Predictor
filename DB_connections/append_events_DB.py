import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

# Load CSV
temp_df = pd.read_csv( 'data/temp_bouts.csv' )

load_dotenv()  # load variables from .env file

# Connect to MySQL
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    database=os.getenv("DB_NAME"),
    password=os.getenv("DB_PASSWORD")
    )
cursor = conn.cursor()

insert_query = """
    INSERT INTO events (
        event_date, event_name, fighter_blue, fighter_red, knockdowns_blue, knockdowns_red,
        sig_attempts_blue, sig_attempts_red, sig_strikes_blue, sig_strikes_red, total_strikes_attempts_blue,
        total_strikes_attempts_red, total_strikes_blue, total_strikes_red, sub_attempts_blue, sub_attempts_red,
        takedowns_blue, takedowns_red, takedown_attempts_blue, takedown_attempts_red, control_time_blue, control_time_red, round, time, weight_class,
        win_method, winner
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
"""

success_count = 0
failed_rows = []

for i, row in temp_df.iterrows():
    if len(row) != 27:
        failed_rows.append((i, row.to_dict(), "Invalid column count"))
        continue
    try:
        cursor.execute(insert_query, tuple(row))
        success_count += 1
    except Exception as e:
        failed_rows.append((i, row.to_dict(), str(e)))  # save row + error

conn.commit()
cursor.close()
conn.close()

print("Finished Updating DB")
print("New Entries added:", success_count)
print("Failed entries:", len(failed_rows))

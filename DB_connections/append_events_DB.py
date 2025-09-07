import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

# Load CSV
temp_df = pd.read_csv('data/temp_bouts.csv')
load_dotenv()

# Connect to MySQL
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    database=os.getenv("DB_NAME"),
    password=os.getenv("DB_PASSWORD")
)
cursor = conn.cursor()

# SQL query - exclude the 'id' column since it's auto-increment
insert_query = """
    INSERT INTO events (
        event_date, event_name, fighter_red, fighter_blue, round, time, weight_class,
        win_method, winner, stance_red, stance_blue, knockdowns_red, knockdowns_blue,
        sig_attempts_red, sig_attempts_blue, sig_strikes_red, sig_strikes_blue,
        total_strikes_attempts_red, total_strikes_attempts_blue, total_strikes_red,
        total_strikes_blue, sub_attempts_red, sub_attempts_blue, takedowns_red,
        takedowns_blue, takedown_attempts_red, takedown_attempts_blue, control_time_red,
        control_time_blue, head_strikes_red, head_strikes_blue, head_attempts_red,
        head_attempts_blue, body_strikes_red, body_strikes_blue, body_attempts_red,
        body_attempts_blue, leg_strikes_red, leg_strikes_blue, leg_attempts_red,
        leg_attempts_blue, distance_red, distance_blue, distance_attempts_red,
        distance_attempts_blue, clinch_strikes_red, clinch_strikes_blue,
        clinch_attempts_red, clinch_attempts_blue, ground_strikes_red,
        ground_strikes_blue, ground_attempts_red, ground_attempts_blue
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
"""

success_count = 0
failed_rows = []
expected_columns = temp_df.shape[1]

print(f"Processing {len(temp_df)} rows...")
print(f"DataFrame has {expected_columns} columns")

for i, row in temp_df.iterrows():
    if len(row) != expected_columns:
        failed_rows.append((i, row.to_dict(), f"Invalid column count: got {len(row)}, expected {expected_columns}"))
        continue
    
    try:
        cursor.execute(insert_query, tuple(row))
        success_count += 1
        if success_count % 5 == 0:  # Progress indicator
            print(f"Processed {success_count} rows successfully...")
    except Exception as e:
        failed_rows.append((i, row.to_dict(), str(e)))
        print(f"Row {i}: FAILED - {str(e)}")

conn.commit()
cursor.close()
conn.close()

print("\n=== FINAL RESULTS ===")
print(f"New Entries added: {success_count}")
print(f"Failed entries: {len(failed_rows)}")

if failed_rows:
    print(f"\nFirst few errors:")
    for i, (row_num, row_data, error) in enumerate(failed_rows[:3]):
        print(f"  Row {row_num}: {error}")
else:
    print("All rows processed successfully!")
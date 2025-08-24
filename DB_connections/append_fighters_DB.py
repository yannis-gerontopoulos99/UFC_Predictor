import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

# Load CSV
temp_df = pd.read_csv('data/temp_fighters.csv')

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
    INSERT INTO fighters (fighter)
    VALUES ( %s)
    """

success_count = 0
failed_rows = []

for i, row in temp_df.iterrows():
    try:
        cursor.execute(insert_query, tuple(row))
        success_count += 1
    except Exception as e:
        failed_rows.append((i, row.to_dict(), str(e)))  # save row + error

conn.commit()
cursor.close()
conn.close()

print("Finished Updating DB")
print("New Entries addded:", len(success_count))
print("Failed entries:", len(failed_rows))

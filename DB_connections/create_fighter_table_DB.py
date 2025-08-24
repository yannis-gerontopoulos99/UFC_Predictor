import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv()  # load variables from .env file

# CONFIGURATION
csv_file_path = 'data/fighters.csv'
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = 3306
db_name = os.getenv("DB_NAME")
table_name = 'fighters'

db_password_enc = urllib.parse.quote_plus(db_password)

# READ CSV
df = pd.read_csv(csv_file_path)

# CREATE DB CONNECTION
engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")


# COLUMN TYPE MAPPING
column_types = {
    "fighter": "VARCHAR(100)"
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
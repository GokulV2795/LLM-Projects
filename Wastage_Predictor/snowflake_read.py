import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def fetch_recent_wastage(days=7):
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )
    sql = f"""
    SELECT * FROM INVENTORY_DB.PUBLIC.WASTAGE_RISK_TABLE
    """
    df = pd.read_sql(sql, ctx)
    ctx.close()
    return df

if __name__ == '__main__':
    print(fetch_recent_wastage(7).head())
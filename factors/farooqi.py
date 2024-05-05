"""
    Resilient Macro-portfolio
    First, take a closer look at traditional defensive stocks. 
    Second, focus on industry themes that don’t depend on economic growth. 
    Third, pay close attention to balance sheets.

    F A C T O R
        Uses latest month FRED https://fred.stlouisfed.org/ 
        open source economic indicators

        Positive:
            Economic (higher)
            1. Real Gross Domestic Product (GDP)
            2. Industrial Production
            Employment (higher)
            3. Nonfarm Payrolls
            4. Unemployment Rate
            Inflation (moderate, set a>=x>=b):
            5. Consumer Price Index (CPI)
            6. Producer Price Index (PPI)
            Interest rates (lower):
            7. Federal Funds Rate
            8. Treasury Yields


"""

from fredapi import Fred
import datetime
import pandas as pd
import os
import logging
import json

DATABASE_DIR = r"C:/Users/papia/OneDrive/Databases/FRED"

LOGGER = logging.getLogger(__name__)

def main():
    """
        Fetch the latest economic indicators from FRED
        Will rerun everyday to get the latest data
    """
    today = datetime.date.today()

    if os.path.exists(f"{DATABASE_DIR}/{str(today)}/economic_indicators.csv"):
        LOGGER.info("Data is up-to-date.")
        df = pd.read_csv(f"{DATABASE_DIR}/{str(today)}/economic_indicators.csv")
        with open(f"{DATABASE_DIR}/{str(today)}/latest_indicators.json", 'r') as f:
            latest = json.load(f)

        return df, latest
    
    os.makedirs(f"{DATABASE_DIR}/{str(today)}", exist_ok=True)

    # You need to register for an API key at the FRED website
    fred = Fred(api_key='4c3e5d4494e935f26c612cca3b83e4ff')

    # Define the series IDs for the indicators
    # You can find these IDs on the FRED website
    series_ids = {
        'Real Gross Domestic Product (GDP)': 'GDPC1',
        'Industrial Production': 'INDPRO',
        'Nonfarm Payrolls': 'PAYEMS',
        'Unemployment Rate': 'UNRATE',
        'Consumer Price Index (CPI)': 'CPIAUCSL',
        'Producer Price Index (PPI)': 'PPIACO',
        'Federal Funds Rate': 'FEDFUNDS',
        'Treasury Yields': 'GS10'
    }

    data = pd.DataFrame()
    latest = series_ids.copy()

    # Fetch the latest values for each series
    for name, series_id in series_ids.items():
        series = fred.get_series(series_id)
        # latest_date = series.index[-1]
        latest_date = series.index[-1].strftime('%Y-%m-%d')  # Convert Timestamp to string
        latest_value = series.iloc[-1]
        print(f"{name} ({latest_date}): {latest_value}")
        #Save the series to a df
        data[name] = series
        latest[name] = [latest_date, latest_value]
        # Save latest as a dictionary in a json file in the same directory
        with open(f"{DATABASE_DIR}/{str(today)}/latest_indicators.json", 'w') as f:
            json.dump(latest, f)

    data.to_csv(f"{DATABASE_DIR}/{str(today)}/economic_indicators.csv")
    return data, latest


if __name__ == '__main__':
    fred_historical, fred_latest = main()
    print(fred_historical.head())
    print(fred_latest)
    
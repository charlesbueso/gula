from bs4 import BeautifulSoup

import logging
import pandas as pd
import pickle
import requests
import time
import os
from datetime import datetime, timedelta
from typing import Any, List, Optional


ROOT = 'https://efdsearch.senate.gov'
LANDING_PAGE_URL = '{}/search/home/'.format(ROOT)
SEARCH_PAGE_URL = '{}/search/'.format(ROOT)
REPORTS_URL = '{}/search/report/data/'.format(ROOT)

BATCH_SIZE = 100
RATE_LIMIT_SECS = 2

PDF_PREFIX = '/search/view/paper/'
LANDING_PAGE_FAIL = 'Failed to fetch filings landing page'

REPORT_COL_NAMES = [
    'tx_date',
    'file_date',
    'last_name',
    'first_name',
    'order_type',
    'ticker',
    'asset_name',
    'tx_amount'
]

DATABASE_DIR = r"C:/Users/papia/OneDrive/Databases/SenatorData"

LOGGER = logging.getLogger(__name__)


def add_rate_limit(f):
    def with_rate_limit(*args, **kw):
        time.sleep(RATE_LIMIT_SECS)
        return f(*args, **kw)
    return with_rate_limit


def _csrf(client: requests.Session) -> str:
    """ Set the session ID and return the CSRF token for this session. """
    landing_page_response = client.get(LANDING_PAGE_URL)
    assert landing_page_response.url == LANDING_PAGE_URL, LANDING_PAGE_FAIL

    landing_page = BeautifulSoup(landing_page_response.text, 'lxml')
    form_csrf = landing_page.find(
        attrs={'name': 'csrfmiddlewaretoken'}
    )['value']
    form_payload = {
        'csrfmiddlewaretoken': form_csrf,
        'prohibition_agreement': '1'
    }
    client.post(LANDING_PAGE_URL,
                data=form_payload,
                headers={'Referer': LANDING_PAGE_URL})

    if 'csrftoken' in client.cookies:
        csrftoken = client.cookies['csrftoken']
    else:
        csrftoken = client.cookies['csrf']
    return csrftoken


def senator_reports(client: requests.Session, date_from) -> List[List[str]]:
    """ Return all results from the periodic transaction reports API. """
    token = _csrf(client)
    idx = 0
    reports = reports_api(client, idx, token, date_from)
    all_reports: List[List[str]] = []
    while len(reports) != 0:
        all_reports.extend(reports)
        idx += BATCH_SIZE
        reports = reports_api(client, idx, token, date_from)
    return all_reports


#'01/01/2024 00:00:00' in end_date for creating pickle with 'outdated' data
def reports_api(
    client: requests.Session,
    offset: int,
    token: str,
    date_from: str
) -> List[List[str]]:
    """ Query the periodic transaction reports API. """
    login_data = {
        'start': str(offset),
        'length': str(BATCH_SIZE),
        'report_types': '[11]',
        'filer_types': '[]',
        'submitted_start_date': date_from,
        'submitted_end_date': '', 
        'candidate_state': '',
        'senator_state': '',
        'office_id': '',
        'first_name': '',
        'last_name': '',
        'csrfmiddlewaretoken': token
    }
    LOGGER.info('Getting rows starting at {}'.format(offset))
    # response = client.post(REPORTS_URL,
    #                        data=login_data,
    #                        headers={'Referer': SEARCH_PAGE_URL})
    # return response.json()['data']
    for _ in range(3):  # Retry up to 3 times
        try:
            response = client.post(REPORTS_URL,
                                   data=login_data,
                                   headers={'Referer': SEARCH_PAGE_URL})
            return response.json()['data']
        except Exception as e:
            LOGGER.error(f"Error occurred: {e}. Retrying in 5 seconds...")
            # LOGGER.error(f"{response.text}")
            time.sleep(5)  # Wait for 5 seconds before retrying

    raise Exception("Failed to get data from API after 3 attempts")


def _tbody_from_link(client: requests.Session, link: str) -> Optional[Any]:
    """
    Return the tbody element containing transactions for this senator.
    Return None if no such tbody element exists.
    """
    report_url = '{0}{1}'.format(ROOT, link)
    report_response = client.get(report_url)
    # If the page is redirected, then the session ID has expired
    if report_response.url == LANDING_PAGE_URL:
        LOGGER.info('Resetting CSRF token and session cookie')
        _csrf(client)
        report_response = client.get(report_url)
    report = BeautifulSoup(report_response.text, 'lxml')
    tbodies = report.find_all('tbody')
    if len(tbodies) == 0:
        return None
    return tbodies[0]


def txs_for_report(client: requests.Session, row: List[str]) -> pd.DataFrame:
    """
    Convert a row from the periodic transaction reports API to a DataFrame
    of transactions.
    """
    first, last, _, link_html, date_received = row
    link = BeautifulSoup(link_html, 'lxml').a.get('href')
    # We cannot parse PDFs
    if link[:len(PDF_PREFIX)] == PDF_PREFIX:
        return pd.DataFrame()

    tbody = _tbody_from_link(client, link)
    if not tbody:
        return pd.DataFrame()

    stocks = []
    for table_row in tbody.find_all('tr'):
        cols = [c.get_text() for c in table_row.find_all('td')]
        tx_date, ticker, asset_name, asset_type, order_type, tx_amount =\
            cols[1], cols[3], cols[4], cols[5], cols[6], cols[7]
        if asset_type != 'Stock' and ticker.strip() in ('--', ''):
            continue
        stocks.append([
            tx_date,
            date_received,
            last,
            first,
            order_type,
            ticker,
            asset_name,
            tx_amount
        ])
    return pd.DataFrame(stocks).rename(
        columns=dict(enumerate(REPORT_COL_NAMES)))

def get_raw_senators_tx(pickle_dir = str):
    if os.path.exists(pickle_dir):
        LOGGER.info(f"Pickle directory exists: {pickle_dir}")
        with open(pickle_dir, 'rb') as f:
            senator_txs_old = pickle.load(f)
        updated_senators = update_database(pickle_dir, senator_txs_old)
        return updated_senators
    else:
        updated_senators = main()
        return updated_senators

def update_database(pickle_dir, senator_txs_old):
    """
    Access '{DATABASE_DIR}/senators.pickle' and get the latest date.
    If the latest date is not the latest available, it will update the data.
    """
    # Convert tx_date column to datetime
    senator_txs_old['file_date'] = pd.to_datetime(senator_txs_old['file_date'], format='%m/%d/%Y', errors='coerce')

    # nat_values = senator_txs_old['file_date'].isna()
    # if nat_values.any():
    #     print("Found NaT values in the tx_date column:")
    #     print(senator_txs_old[nat_values])

    latest_date = senator_txs_old['file_date'].max()
    days_padding = 5

    # Compare max date in picke file with today's date
    LOGGER.info(f"Today's date: {datetime.today()}")
    LOGGER.info(f"Latest date in pickle file: {str(latest_date)}")

    # Convert back to string date
    senator_txs_old['file_date'] = senator_txs_old['file_date'].dt.strftime('%m/%d/%Y')

    if (latest_date + timedelta(days=days_padding)) < datetime.today():
        # TODO: Update the pickle file with the new transactions
        LOGGER.info("Updating pickle file")
        updated_database = main(pickle_dir=pickle_dir, date_from=latest_date, senator_txs_old=senator_txs_old)
        return updated_database
    else:
        LOGGER.info("No need to update pickle file, up-to-date.")
        return senator_txs_old


def main(pickle_dir=f"{DATABASE_DIR}/senators.pickle",
         csv_dir=f"{DATABASE_DIR}/senators.csv", 
         date_from='11/01/2023 00:00:00', 
         senator_txs_old=pd.DataFrame()) -> pd.DataFrame:
    """
        Full run from date_from to present
        Checks all reports and saves/updates the database
        in pickle_dir, both in pickle and csv format

        'senator_txs_old' parameter allows dynamic updating of database
    """

    LOGGER.info('Initializing client')
    client = requests.Session()
    client.get = add_rate_limit(client.get)
    client.post = add_rate_limit(client.post)
    reports = senator_reports(client, date_from=date_from)
    all_txs = senator_txs_old
    for i, row in enumerate(reports):
        if i % 10 == 0:
            LOGGER.info('Fetching report #{}'.format(i))
            LOGGER.info('{} transactions total'.format(len(all_txs)))
        txs = txs_for_report(client, row)
        all_txs = all_txs._append(txs)

    if all_txs.equals(senator_txs_old):
        LOGGER.info('No new transactions found')
        return all_txs
    else:
        LOGGER.info('New transactions found')
        LOGGER.info('Dumping to .pickle and csv')
        with open(pickle_dir, 'wb') as f:
            pickle.dump(all_txs, f)
        LOGGER.info('Uploaded new pickle!')
        # Also dump in same directory as csv
        all_txs.to_csv(csv_dir)
        LOGGER.info('Uploaded new csv!')
        return all_txs




if __name__ == '__main__':
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Use stored database
    pickle_dir = f"{DATABASE_DIR}/senators.pickle"
    senators_txs = get_raw_senators_tx(pickle_dir)
    # with open(pickle_dir, 'rb') as f:
    #     senators_txs = pickle.load(f)

    # Print information about senators_txs df
    LOGGER.info(str(senators_txs.info()))
    LOGGER.info(str(senators_txs.head()))
    LOGGER.info(str(senators_txs.tail()))
    
    # print all rows of senators_txs with LOGGER
    # for index, row in senators_txs.iterrows():
    #     LOGGER.info(row)


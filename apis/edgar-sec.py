from nyse_tickers import nyse_tickers
from sec_edgar_downloader import Downloader

"""
    Gets last X number of 10Q filings from SEC website
    for each given ticker
"""

def company_10q_filings(tickers=['msft'], limit=1):
  """
    Get 'limit' amount of 10-Q filings
    for each ticker in 'tickers' list

    Output:
    Saves filings in "C:/repos/gula/database/sec_edgar_filings/top5/"
  """
  dl = Downloader("Hobby", "papia1999@hotmail.com", "C:/repos/gula/database/sec_edgar_filings/top5/")
  
  # Get the latest five 10-Q filings for Microsoft
  for ticker in tickers:
    dl.get("10-Q", ticker, limit=limit, download_details=True)

  return dl


if __name__ == "__main__":

  tickers = ['msft']
  print(company_10q_filings(tickers=tickers, limit=5))
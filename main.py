from universe.nyse.nyse_tickers import nyse_tickers

print(nyse_tickers[:5])

"""
    First we run altman and johnfkennedy on all the stocks in the NYSE.
    We look for highest scoring stocks.
    We match with current S&P500 stocks, and buy X stocks equally weighted.
"""
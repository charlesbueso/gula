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
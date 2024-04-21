import html2text

def html_parser(html_filepath, text_filepath):

    # Read the text file
    with open(html_filepath, "r", encoding="utf-8") as f:
        text_content = f.read() 

    # Parse HTML
    cleaned_text = html2text.html2text(text_content)
    with open(text_filepath, "w", encoding="utf-8") as f:
        f.write(cleaned_text)



if __name__ == "__main__":
    html_parser("""C:/repos\gula\database\sec_edgar_filings/top5\sec-edgar-filings\MSFT/10-K/0000950170-23-035122\primary-document.html""", 
                "C:/repos/gula/database/sec_edgar_filings/top5/23msft.txt")
    html_parser("C:/repos\gula\database\sec_edgar_filings/top5\sec-edgar-filings\MSFT/10-K/0001564590-22-026876\primary-document.html", 
            "C:/repos/gula/database/sec_edgar_filings/top5/22msft.txt")
    html_parser("C:/repos\gula\database\sec_edgar_filings/top5\sec-edgar-filings\MSFT/10-K/0001564590-21-039151\primary-document.html", 
            "C:/repos/gula/database/sec_edgar_filings/top5/21msft.txt")
    html_parser("C:/repos\gula\database\sec_edgar_filings/top5\sec-edgar-filings\MSFT/10-K/0001564590-20-034944\primary-document.html", 
            "C:/repos/gula/database/sec_edgar_filings/top5/20msft.txt")
    html_parser("C:/repos\gula\database\sec_edgar_filings/top5\sec-edgar-filings\MSFT/10-K/0001564590-19-027952\primary-document.html", 
            "C:/repos/gula/database/sec_edgar_filings/top5/19msft.txt")
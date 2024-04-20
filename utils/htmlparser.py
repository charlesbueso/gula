from bs4 import BeautifulSoup
import re

def html_parser(html_filepath, text_filepath):

    # Read the text file
    with open(html_filepath, "r") as f:
        text_content = f.read() 

    # Parse HTML
    soup = BeautifulSoup(text_content, "html.parser")

    # Extract text and remove formatting (optional)
    cleaned_text = soup.get_text(separator="\n")  # Use "\n" for newlines after each element
    cleaned_text = re.sub(r'[^\w\s\d\.]', '', cleaned_text)  # Optional: remove special characters (adjust regex for specific needs)

    # Print or save the cleaned text
    print(cleaned_text)
    # or
    with open(text_filepath, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
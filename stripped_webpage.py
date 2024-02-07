import requests
from bs4 import BeautifulSoup

def strip_webpage(url):
    try:
        # Fetch the HTML content of the webpage
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unnecessary elements (scripts and styles)
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Extract text content including table content
        text_content = []
        for element in soup.recursiveChildGenerator():
            if isinstance(element, str):
                text_content.append(element)
            elif element.name == 'table':
                # Handle table content
                table_text = extract_table_content(element)
                text_content.append(table_text)

        return '\n'.join(text_content)

    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_table_content(table):
    table_content = []
    for row in table.find_all('tr'):
        row_content = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
        row_text = '\t'.join(row_content)
        table_content.append(row_text)
    return '\n'.join(table_content)

# Example usage
url = 'https://www.topendsports.com/sport/soccer/list-player-of-the-year-ballondor.htm'
stripped_content = strip_webpage(url)

if stripped_content:
    print(stripped_content)

import requests
from bs4 import BeautifulSoup
import json

URL = "https://www.bbc.com/business"

page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

results = soup.find(id = "__NEXT_DATA__", type = "application/json" )

json_data = json.loads(results.text)

# Navigate through the JSON data to find the title
#title = json_data['props']['pageProps']['page']['@"business",']['sections'][0]['content'][0]['title']

#print(title)

# Initialize a list to store the titles
titles = []

# Function to recursively extract titles from the JSON data
def extract_titles(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'title':
                titles.append(value)
            else:
                extract_titles(value)
    elif isinstance(data, list):
        for item in data:
            extract_titles(item)

# Extract titles from the JSON data
extract_titles(json_data)

index = next((i for i, element in enumerate(titles) if "More in Business" in element), None)

# Remove all elements after and including the element that contains "More in Business"
if index is not None:
    elements = titles[:index]


# Print all the titles
for title in elements:
    print(title)
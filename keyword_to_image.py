import requests
from urllib.parse import urljoin
import requests

def fetch_image_url(query, api_key):
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {'Ocp-Apim-Subscription-Key': api_key}

    params = {
        'q': query,
        'count': 1  # Number of images to fetch
    }

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if 'value' in data and data['value']:
            image_url = data['value'][0]['contentUrl']
            return image_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
    return None

# # Example usage:
api_key = ''
# toy_name = "Marble Roller Coaster"
# image_url = fetch_image_url(toy_name, api_key)
# print(image_url)

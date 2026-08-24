import os
import json

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TMDB_API_KEY")

url = "https://api.themoviedb.org/3/movie/popular"

params = {
    "api_key": api_key,
    "language": "en-US",
    "page": 1,
}

response = requests.get(
    url,
    params=params,
    timeout=10,
)

response.raise_for_status()

data = response.json()

movie = data["results"][0]

print(json.dumps(movie, indent=4))
import requests


def fetch_latest_headlines(api_key):
    """Fetch top headlines from NewsAPI. Returns safe fallback data on failure."""
    if not api_key:
        return [
            {
                "title": "Demo headline: Add NEWS_API_KEY to fetch live headlines.",
                "description": "Set your API key in environment variables for real-time integration.",
                "url": "",
                "source": {"name": "System"},
            }
        ]

    endpoint = "https://newsapi.org/v2/top-headlines"
    params = {"country": "us", "pageSize": 12, "apiKey": api_key}
    try:
        response = requests.get(endpoint, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        return payload.get("articles", [])
    except Exception:
        return [
            {
                "title": "Unable to fetch live headlines right now.",
                "description": "Please try again later or check your NEWS_API_KEY configuration.",
                "url": "",
                "source": {"name": "System"},
            }
        ]

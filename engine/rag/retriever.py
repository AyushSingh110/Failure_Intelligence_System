import requests

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def fetch_wikipedia_summary(query: str) -> str:
    """
    Fetches summary from Wikipedia for a given query.
    """

    try:
        url = WIKI_API + query.replace(" ", "_")
        response = requests.get(url)

        if response.status_code != 200:
            return ""

        data = response.json()

        return data.get("extract", "")

    except Exception:
        return ""
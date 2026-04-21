import json

def load_movies() -> list[dict]:
    with open('data/movies.json') as f:
        data = json.load(f)
    return data['movies']

def load_stopwords() -> list[str]:
    with open('data/stopwords.txt') as f:
        return [w for w in f.read().splitlines() if w]

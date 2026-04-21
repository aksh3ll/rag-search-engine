import string
from nltk.stem import PorterStemmer
from data_utils import load_stopwords

translation_table = str.maketrans("", "", string.punctuation)
stopwords = load_stopwords()
stemmer = PorterStemmer()

def to_tokens(text: str) -> list:
    return [
        stemmer.stem(w)
        for w in text.lower().translate(translation_table).split()
        if w and w not in stopwords
    ]


from nltk.stem import SnowballStemmer

def steam_text(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    return " ".join(stemmed_words)
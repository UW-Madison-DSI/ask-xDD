from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download("punkt")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text: str) -> List[str]:
    """Generic text cleaning function."""

    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    words = [
        [word.lower() for word in sentence if word not in string.punctuation]
        for sentence in words
    ]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [
        [word for word in sentence if word not in stop_words] for sentence in words
    ]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in words]

    return [" ".join(sentence) for sentence in words]

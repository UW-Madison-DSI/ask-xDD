from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def to_sentences(text: str) -> List[str]:
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


def to_chunks(sentences: List[str], n=5000) -> List[str]:
    """Concatenate sentences into chunks with around n characters."""
    chunks = []

    for i, sentence in enumerate(sentences):
        if i == 0:
            chunks.append(sentence)
        else:
            if len(chunks[-1]) + len(sentence) < n:
                chunks[-1] += "\n" + sentence
            else:
                chunks.append(sentence)

    return chunks

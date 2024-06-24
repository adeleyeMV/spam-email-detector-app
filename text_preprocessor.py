import re
import string
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def expand_contractions(self, text):
        return contractions.fix(text, slang=False)

    def clean_words(self, text):
        text = re.sub(r"[^a-zA-Z ]+", " ", text.strip())
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub('<.*?>', '', text)
        text = ''.join([char for char in text if char not in string.punctuation])
        text = text.replace('\n', ' ')
        return text

    def tokenize(self, text):
        lowercase = text.lower()
        tokens = word_tokenize(lowercase)
        return " ".join(tokens).strip()

    def remove_stopwords(self, text):
        word_list = [word for word in text.split() if word not in self.stopwords]
        return " ".join(word_list)

    def lemmatize(self, text):
        lemmatized = [self.lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(lemmatized)

    def preprocess(self, text):
        text = self.expand_contractions(text)
        text = self.clean_words(text)
        text = self.tokenize(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text

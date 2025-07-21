import os
import re
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    # Lemmatize each word in the text
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    # Remove stop words from the text
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    # Remove all digits from the text
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    # Convert all words in the text to lowercase
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    # Remove punctuations and extra whitespace from the text
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    # Remove URLs from the text
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    # Set text to NaN if sentence has fewer than 3 words
    df.loc[df.text.str.split().str.len() < 3, 'text'] = np.nan

def normalize_text(df):
    # Apply all preprocessing steps to the 'content' column of the DataFrame
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    return df

def normalized_sentence(sentence):
    # Apply all preprocessing steps to a single sentence
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

# Load raw train and test data
train_data = pd.read_csv("data/raw/train.csv")
test_data = pd.read_csv("data/raw/test.csv")

# Normalize train and test data
train_data = normalize_text(train_data)
test_data = normalize_text(test_data)

# Save processed data to CSV files
os.makedirs("data/processed", exist_ok=True)  # Ensure the directory exists
train_data.to_csv("data/processed/train.csv", index=False)
test_data.to_csv("data/processed/test.csv", index=False)

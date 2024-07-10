import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

# Loading Dataset

df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

df.info()

# Initialize stemmer and stopwords

stemmer = PorterStemmer()
corpus = []

stopwords_set = set(stopwords.words('english'))

# Preprocessing the text data

for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

# Vectorize the text data

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the classifier

clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

# Display the accuracy of the classifier

print(f'Accuracy: {clf.score(X_test, y_test)}')


# Function to preprocess and classify a new email
def classify_email(email):
    email_text = email.lower().translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    email_text = ' '.join(email_text)

    # Vectorize the email text

    email_corpus = [email_text]
    X_email = vectorizer.transform(email_corpus)

    # Predict the label for the email
    prediction = clf.predict(X_email)

    # Print the result
    if prediction == 1:
        print('This Email Has been Labeled as Spam')
    else:
        print('This Email appears legitimate')


# Email from the dataset

email_to_classify = df.text.values[10 ]
classify_email(email_to_classify)

# External email

external_email = "Spencer Since you’ve been gone, we’ve added thousands of new restaurants to Uber Eats. Whether it's breakfast, lunch, dinner, or a late night snack, we have what you're craving. Give us another try with 60% off your next 2 orders through Sunday. Max savings of $20. Order now Save on today’s treatsThese restaurants have offers on tasty menu items."
classify_email(external_email)

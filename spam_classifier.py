import streamlit as st
import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
  
class SpamClassifier:
    def __init__(self):
        self.model = MultinomialNB()
        self.vec = CountVectorizer(encoding='latin-1', stop_words='english')
        df = pd.read_csv('Spam_Filter/sms_spam/spam.csv', encoding='latin-1')
        data = df[['v2', 'v1']].rename(columns={'v2': 'text', 'v1': 'label'})
        enc = LabelEncoder()
        data['label'] = enc.fit_transform(data['label'])
        X = data.drop(columns=['label'])["text"]
        Y = data['label']

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        self.vec.fit(x_train)
        x_train = self.vec.transform(x_train).toarray()
        x_test = self.vec.transform(x_test).toarray()
        self.model.fit(x_train, y_train)

    def predict(self, message):
        X = self.vec.transform(message).toarray()
        prediction = self.model.predict(X)
        return prediction

st.title("Spam e-mail Filter")

classifier = SpamClassifier()

message = st.text_input("Enter Your e-mail", "")
btn = st.button("Predict")

if btn:
  out = None
  out = classifier.predict([message])

  if int(out[0]) == 1:
    st.subheader(f"The e-mail is a spam")
  else:
    st.subheader(f"This e-mail is not a spam")

else:
   st.write("Enter an SMS and it will predict whether it is spam or not.")
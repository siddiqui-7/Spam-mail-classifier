import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from CSV file
@st.cache
def load_data():
    raw_mail_data = pd.read_csv('mail_data.csv')
    mail_data = raw_mail_data.fillna('')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    return mail_data

mail_data = load_data()

# Separate the data as text and label
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Initialize the TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform the training data
X_train_features = feature_extraction.fit_transform(X_train)

# Transform the test data
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Streamlit app
st.title("Spam/Ham Mail Classifier")

# Input for user to enter mail text
input_mail = st.text_area("Enter the mail text:")

if st.button("Predict"):
    if input_mail:
        # Convert text to feature vector
        input_data_features = feature_extraction.transform([input_mail])
        
        # Making prediction
        prediction = model.predict(input_data_features)
        
        # Print result based on prediction
        if prediction[0] == 1:
            st.write('Ham Mail')
        else:
            st.write('Spam Mail')
    else:
        st.write("Please enter some text to classify.")

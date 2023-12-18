import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


# load data
df = pd.read_csv("creditcard.csv")

# seperate legitimate and fraudulent transaction
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

# undersample legitimate transaction  to balance the classes
legit_sample = legit.sample(n=492, random_state=2)
df = pd.concat([legit_sample, fraud])

# spliting datasets
x = df.drop(columns="Class", axis=1)
y = df["Class"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)

# train logistic model
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy = accuracy_score(lr.predict(x_train), y_train)
precision = precision_score(lr.predict(x_train), y_train)
recall = recall_score(lr.predict(x_train), y_train)
f1 = f1_score(lr.predict(x_train), y_train)


# web app

st.title("Credit Card Fraud Detection")
input_df = st.text_input("Enter All Required Feature Value ")
input_df_splited = input_df.split(",")

submit = st.button("Submit")

if submit:
    feature = np.asarray(input_df_splited, dtype=np.float64)
    prediction = lr.predict(feature.reshape(1, -1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")

    else:
        st.write("fradulant Transcation")

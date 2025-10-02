import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='latin-1')
df

df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df

df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df

new_column_order = ['text', 'label']
df = df[new_column_order]
df


df.info

df["label"].value_counts()

df.isnull().sum()

# Load dataset  
X = df['text']
y = df['label']

df2 = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    df2.append(text)

df2

# creating bag of words using countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df2).toarray()

X.shape

# Split data  
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


# Vectorize text  
vectorizer = TfidfVectorizer(stop_words='english')  
X_train_vec = vectorizer.fit_transform(X_train.astype(str))  
X_test_vec = vectorizer.transform(X_test.astype(str))

# Train model  
model = MultinomialNB()  
model.fit(X_train_vec, y_train)


# Predict & Evaluate
y_pred = model.predict(X_test_vec)  

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

X_test

# visualising the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

### Create a Pickle file using serialization 
import pickle
pickle_out = open("naive_bayes_3.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()













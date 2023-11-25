# -*- coding: utf-8 -*-
"""SentimentAnalysisForSocialMediaData(Twitter).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WsFVIbyo6rdRrvfEvYpCwsgD8JVQkUXq

# Importing the Dependencies
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
import time

import nltk
nltk.download('stopwords')

"""# **1.Data Collection**"""

!pip install kaggle

"""Upload the kaggle json file"""

# configure the path of kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

"""### Importing Twitter Sentiment dataset"""

# API to fetch the dataset from Kaggle
!kaggle datasets download -d kazanova/sentiment140

# Extracting the compressed dataset

from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print("The dataset is extracted")

"""# **2.Text Preprocessing**"""

# loading the data from csv file to pandas dataframe
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

# Checking the number of rows and columns
twitter_data.shape

twitter_data.head()

# naming the columns and reading the dataset again

column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', names=column_names)

twitter_data.shape

twitter_data.head()

twitter_data['target'].unique()

# counting the number of missing values in the dataset
twitter_data.isnull().sum()

# checking the distribution of target column
twitter_data['target'].value_counts()

"""### Note: The data is equally distributed i.e the distribution is even, if the distribution is not even we have to do up-sampling and downn-sampling to make it even

COnvert the target code from 4 to 1
"""

twitter_data.replace({'target':{4:1}}, inplace=True)

twitter_data['target'].value_counts()

"""0 --> Negative Tweet


1 --> Positive Tweet

**Stemming**

Stemming is the process of reducing a word to its Root word
"""

stemmer = PorterStemmer()

def stemming(content):
  # remove twitter handles
  r = re.findall("@[\w]*", content)
  stemmed_content = [re.sub(word, '', content) for word in r]
  # remove special characters, numbers and punctuation
  stemmed_content = re.sub('[^a-zA-Z#]', ' ', content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [stemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

start = time.time()
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
end = time.time()

print("Time taken: ", end-start)

twitter_data.to_csv("/content/sample_data/twitter_data.csv", sep=',', index=False, encoding='ISO-8859-1')

twitter_data.head()

print(twitter_data['stemmed_content'])

print(twitter_data['target'])

"""## **Exploratory Data Analysis**"""



! pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

all_words = " ".join([sentence for sentence in twitter_data['stemmed_content']])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)


# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Frequent words visualization for +ve
pos_words = " ".join([sentence for sentence in twitter_data['stemmed_content'][twitter_data['target']==1]])
pos_wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(pos_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Frequent words visualization for -ve
neg_words = " ".join([sentence for sentence in twitter_data['stemmed_content'][twitter_data['target']==0]])
neg_wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(neg_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



# separating the data and label
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

print(X)

print(Y
)

"""# **3.Feature Extraction**

## Splitting the data to training data and test data
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)

print(X_test)

# convert the textual data to numerical data

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train)

print(X_test)

"""# **4.Model Selection**"""

lr_model = LogisticRegression(max_iter=1000)

"""# **5.Model Training**

Training the MAchine Learning Model
"""

lr_model.fit(X_train,Y_train)

"""# **6.Model Evaluation**

Model Evaluation

Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = lr_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
training_data_f1 = f1_score(Y_train,X_train_prediction)
training_data_recall = recall_score(Y_train,X_train_prediction)

print("Accuracy score on the training data : ", training_data_accuracy)
print("F1 score on the training data : ", training_data_f1)
print("Recall score on the training data : ", training_data_recall)

# accuracy score on the test data
X_test_prediction = lr_model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test,X_test_prediction)
testing_data_f1 = f1_score(Y_test,X_test_prediction)
testing_data_recall = recall_score(Y_test,X_test_prediction)

print("Accuracy score on the testing data : ", testing_data_accuracy)
print("F1 score on the training data : ", testing_data_f1)
print("Recall score on the training data : ", testing_data_recall)

"""Saving the model"""

import pickle
file_path = '/content/drive/MyDrive/CodingRaja Internship/Sentiment Analysis/lr_model.pkl'
pickle.dump(lr_model, open(file_path, 'wb'))

"""# **7.Prediction**

Using the saved model for future predictions
"""

# loading the saved model
loaded_lr_model = pickle.load(open('/content/drive/MyDrive/CodingRaja Internship/Sentiment Analysis/lr_model.pkl', 'rb'))

X_new = X_test[3]
print(Y_test[3])

prediction = loaded_lr_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print("Negative Tweet")
else:
  print("Positive Tweet")













"""# Using Different Models

SVM
"""

from sklearn import svm
svm_model = svm.SVC(max_iter=1000)

svm_model.fit(X_train,Y_train)

"""Accuracy Score"""

# accuracy score on the training data
X_train_prediction = svm_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)

print("Accuracy score on the training data : ", training_data_accuracy)

# accuracy score on the test data
X_test_prediction = svm_model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test,X_test_prediction)

print("Accuracy score on the testing data : ", testing_data_accuracy)


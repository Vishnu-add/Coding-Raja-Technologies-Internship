from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import datetime
import os
import pickle
from datapreprocessing.datapreprocessing import data_cleaning
from sklearn.feature_extraction.text import TfidfVectorizer
from datapreprocessing.datapreprocessing import StemTokenizer
import yaml
def load_config(file_path="config.yaml"):
    with open(file_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
    return config_data

config = load_config()
model_folder_path = config.get('model_folder_path')[0]['path']
vectorizer_path = config.get('vectorizer_path')[0]['path']
dataset_path = config.get('dataset_path')[0]['path']


# vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = StemTokenizer(), ngram_range=(1,3),min_df=10,max_features=10000)
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

app = Flask(__name__)
#model_path = os.getcwd()+r'\sentimentanalysis\models\model'

# classifier = joblib.load(model_path+r'/classifier.pkl') 
classifier = pickle.load(open(model_folder_path+r'lr_model.pkl', 'rb'))

def predictfunc(content):    
     cleaned_content = data_cleaning(content)
     tweet = vectorizer.transform([cleaned_content])
     # print(type(tweet))
     prediction = classifier.predict(tweet)
     print(prediction)
     if prediction[0]==1:
          sentiment='Positive'
     else:
          sentiment='Negative'      
     return prediction[0],sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
     
     if request.method == 'POST':
        result = request.form
        content = request.form['tweet']
        prediction,sentiment =predictfunc(content)      
     return render_template("predict.html",pred=prediction,sent=sentiment)

if __name__ == '__main__':
     #app.run(debug = True,port=8080)
     app.run(host='0.0.0.0',debug=True)
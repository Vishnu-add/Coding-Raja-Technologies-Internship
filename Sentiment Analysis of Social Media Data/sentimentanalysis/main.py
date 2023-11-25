import sys
import os
import joblib
import datapreprocessing.datapreprocessing as dp
from datapreprocessing.datapreprocessing import DataCleaning
from datapreprocessing.datapreprocessing import StemTokenizer
from evaluation.evaluationmetrics import precision_score_plot, confusion_matrix_plot
from dataloader.dataload import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import time
import pickle
import yaml
def load_config(file_path="config.yaml"):
    with open(file_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
    return config_data

config = load_config()
model_folder_path = config.get('model_folder_path')[0]['path']
vectorizer_path = config.get('vectorizer_path')[0]['path']
dataset_path = config.get('dataset_path')[0]['path']


#Loading of data
data=load_dataset(dataset_path)

#Split data
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.01, random_state=2)


#Text classifier pipeline   
text_clf = Pipeline(steps=[('clean', DataCleaning()),
('vect',TfidfVectorizer(analyzer = "word", tokenizer = StemTokenizer(), ngram_range=(1,3),min_df=10,max_features=10000)),
('clf',LogisticRegression(penalty='l2',dual=False, tol=0.0001, C=1.0, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None))])

#Train text classifier by using pipeline 
start= time.time()
text_clf.fit(x_train,y_train)
end = time.time()
print("Time taken to train text classifier: ", end-start)

#Generate prediction on test data
y_predict=text_clf.predict(x_test)
y_score = text_clf.predict_proba(x_test)[:, 1]

#Evaluation of base model
print("Precision Score on test dateset for Logistic Regression: %s" % precision_score(y_test,y_predict,average='micro'))
print("AUC Score on test dateset for Logistic Regression: %s" % roc_auc_score(y_test,y_score,multi_class='ovo',average='macro'))
f1_score_train_1 =f1_score(y_test,y_predict,average="weighted")
print("F1 Score test dateset for Logistic Regression: %s" % f1_score_train_1)
confusion_matrix_plot(y_test, y_predict)


#Store base model
# joblib.dump(text_clf, r'D:\CodingRaja Projects\Sentiment Analysis of Social Media Data\Sentiment_Analysis_Case_Study\sentimentanalysis\models\model\lr_classifier.pkl',compress=True)
pickle.dump(text_clf, open(model_folder_path+r"lr_classifier.pkl", 'wb'))
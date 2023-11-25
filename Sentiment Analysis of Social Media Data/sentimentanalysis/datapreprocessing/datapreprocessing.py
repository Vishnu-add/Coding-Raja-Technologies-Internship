#Customize stopword as per data
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
import re
stop_words = stopwords.words('english')
stop_words=set(stop_words)
stemmer = PorterStemmer()

def clean_text(content):
    # remove twitter handles
    r = re.findall("@[\w]*", content)
    stemmed_content = [re.sub(word, '', content) for word in r]
    stemmed_content = ' '.join(stemmed_content)
    # removing URL's
    stemmed_content = re.sub(r'http\S+', ' ', stemmed_content) 

    # remove special characters, numbers and punctuation
    stemmed_content = re.sub('[^a-zA-Z#]', ' ', stemmed_content) 
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

#Data preprocessing
def data_cleaning(content):
    content = clean_text(content)
    return content

class DataCleaning(BaseEstimator,TransformerMixin):
    def __init__(self):
        print('calling--init--')
    def fit(self,X,y=None):
        print('calling fit')
        return self
    def transform(self, X,y=None):
        print('calling transform')
        X=X.apply(data_cleaning)
        return X

# Stemming of word 
class StemTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, tweet):
        return [self.stemmer.stem(word) for word in word_tokenize(tweet)]
    
    
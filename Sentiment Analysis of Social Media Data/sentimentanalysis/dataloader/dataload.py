import pandas as pd

def load_dataset(path):
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(path,encoding='ISO-8859-1',  names=column_names)
    data=df[['text','target']]
    return pd.concat([data[:10], data[1599990:]])
    # return data
    
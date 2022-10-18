import sklearn
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd
import csv
import string
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def read_text(wd):

    for i in range(1, 6): #6
        # Change the directory
        path= wd + "\\fold" + str(i)
        os.chdir(path)
        # iterate through all file
        for idx, file in enumerate(os.listdir()):
            # Check whether file is in text format or not
            file_path = f"{path}\\{file}"
           
            if idx==0 and i==1 :
                X=pd.read_csv(file_path, header=None,sep='\s\s+', quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=False, engine="python")
            else:
                X=pd.concat([X,pd.read_csv(file_path, header=None,sep='\s\s+', quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=False, engine="python")])
    X.columns=['review']
    return X


def preprocess_sentence(text: str) -> list:
    # Remove punctuation
    text="".join([i for i in text if i not in string.punctuation])
    # Remove capital letters
    text = text.lower()
    # remove numbers
    text=''.join([i for i in text if not i.isdigit()])
    
    # Tokenize words
    tokens = re.split(r'\s+',text)
    # lemmatize words
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    return lemm_text


# Folder Path
path = os.path.join(os.getcwd(), "negative_polarity")
folder_list=["\\deceptive_from_MTurk", "\\truthful_from_Web"]

#deceptive
X=read_text(path+folder_list[0])
X_test=X.iloc[-80:,:]
X_train=X.iloc[:-80,:]

# deceptive + truthful
X1=read_text(path+folder_list[1])
X_test=pd.concat([X_test,X1.iloc[-80:,:]])
X_train=pd.concat([X_train,X1.iloc[:-80,:]])
y_test=[1]*80 + [0]*80
y_train=[1]*320 + [0]*320

# (1) remove punctuation, (2) to lower (3) remove numbers (4) remove unnecessary white spaces (5) stopwords (6) lemmatization
#stopwords
stop=stopwords.words('english')
vector=CountVectorizer(stop_words=stop).fit(pd.concat([X,X1]))
print(type(vector))






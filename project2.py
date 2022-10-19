import sklearn
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd
import csv
import string
import re
import nltk

# from itertools import islice, izip
# from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

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

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_sentence(data: pd.DataFrame) -> pd.DataFrame:
    # Remove punctuation
    data['review'] = data['review'].apply(lambda text: "".join([i for i in text if i not in string.punctuation]))
    # Remove capital letters
    data['review'] = data['review'].apply(lambda text: text.lower())
    # Remove numbers
    data['review'] = data['review'].apply(lambda text: ''.join([i for i in text if not i.isdigit()]))
    
    # Tokenize words
    data['review'] = data['review'].apply(lambda text: re.split(r'\s+',text))


    # Lemmatize words
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    data['review'] = data['review'].apply(lambda text: ' '.join([wordnet_lemmatizer.lemmatize(word, [nltk.pos_tag(word)[1]]) for word in text]))
    return data

# Download NLP corpora
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Folder Path
path = os.path.join(os.getcwd(), "negative_polarity")
folder_list = ["\\deceptive_from_MTurk", "\\truthful_from_Web"]

# Collect deceptive (1) and truthful (1) reviews
data = pd.concat([read_text(path+folder_list[0]), read_text(path+folder_list[1])])
data['class'] = [0]*400 + [1]*400
prep = preprocess_sentence(data)

# Vectorize the data excluding stopwords
stop = stopwords.words('english')
vector = CountVectorizer(stop_words=stop).fit_transform(prep['review'])

# Select the first five rows from the data set
td = pd.DataFrame(vector.todense()).iloc[:5]
td.columns = vector.get_feature_names()
term_document_matrix = td.T
term_document_matrix.columns = ['Rev ' + str(i) for i in range(1, 6)]
term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

# Top 25 words
term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False)[:25]

# Print the first 10 rows
print(term_document_matrix.drop(columns=['total_count']).head(10))
term_document_matrix['total_count'].plot.bar()


# X_test=X1.iloc[-80:,:]
# X_train=X1.iloc[:-80,:]
#
#
# X_test=pd.concat([X_test,X2.iloc[-80:,:]])
# X_train=pd.concat([X_train,X2.iloc[:-80,:]])
# y_test=[1]*80 + [0]*80
# y_train=[1]*320 + [0]*320

# # bigrams
# words = Counter()
# for idx, sentence in enumerate(pd.concat([X, X1]):
#
#     words = words + re.findall("\w+", sentence)
#     print (Counter(izip(words, islice(words, 1, None))))




# (1) remove punctuation, (2) to lower (3) remove numbers (4) remove unnecessary white spaces (5) stopwords (6) lemmatization
#stopwords
stop=stopwords.words('english')
vector=CountVectorizer(stop_words=stop).fit_transform(pd.concat([X1,X2]))
#vector= vector.transform
print(type(vector))








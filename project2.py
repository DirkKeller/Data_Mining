import sklearn
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd
import csv
import string
import re
import nltk


# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet

# def read_text(wd):

#     for i in range(1, 5): #6
#         # Change the directory
#         path= wd + "\\fold" + str(i)
#         os.chdir(path)
#         # iterate through all file
#         for idx, file in enumerate(os.listdir()):
#             # Check whether file is in text format or not
#             file_path = f"{path}\\{file}"
           
#             if idx==0 and i==1 :
#                 X=pd.read_csv(file_path, header=None,sep='\s\s+', quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=False, engine="python")
#             else:
#                 X=pd.concat([X,pd.read_csv(file_path, header=None,sep='\s\s+', quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=False, engine="python")])
#     X.columns=['review']
#     return X

# def preprocess_sentence(data: pd.DataFrame) -> pd.DataFrame:
#     # Remove punctuation
#     data['review'] = data['review'].apply(lambda text: "".join([i for i in text if i not in string.punctuation]))
#     # Remove capital letters
#     data['review'] = data['review'].apply(lambda text: text.lower())
#     # Remove numbers
#     data['review'] = data['review'].apply(lambda text: ''.join([i for i in text if not i.isdigit()]))
    
#     # Tokenize words
#     data['review'] = data['review'].apply(lambda text: re.split(r'\s+',text))


#     # Lemmatize words
#     word_lemmatizer=nltk.stem.WordNetLemmatizer()
#     data['review'] = data['review'].apply(lambda text: ' '.join([word_lemmatizer.lemmatize(word) for word in text]))  # , [get_wordnet_pos(word)]
#     return data

# # Download NLP corpora
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

# # Folder Path
# path = os.path.join(os.getcwd(), "negative_polarity")
# folder_list = ["\\deceptive_from_MTurk", "\\truthful_from_Web"]

# # Collect deceptive (0) and truthful (1) reviews
# data = pd.concat([read_text(path+folder_list[0]), read_text(path+folder_list[1])])
# prep = preprocess_sentence(data)

# # Vectorize the data excluding stopwords
# stop = stopwords.words('english')

# ngram_range = (1,2)
# max_features = 1000
# use_idf = True

# vector = TfidfVectorizer(max_features=max_features,
#                         ngram_range=ngram_range,
#                         min_df=1, 
#                         max_df=1.0,
#                         use_idf=use_idf)

# vec=vector.fit_transform(prep['review'])
# df = pd.DataFrame(vec.toarray().transpose(),index=vector.get_feature_names())
# df.columns=['Rev ' + str(i) for i, _ in enumerate(df.columns)]
# df = df.T
# df['class'] = [0]*320 + [1]*320
# os.chdir(path)
# df.to_csv('reviews_document_term.csv', index=False)
# term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

# # Top 25 words
# term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False)[:25]

# # Print the first 10 rows
# print(term_document_matrix.drop(columns=['total_count']).head(10))
# term_document_matrix['total_count'].plot.bar()

# simulate splitting a dataset of 25 observations into 5 folds
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle = False).split(range(25))

# k_range = list(range(1, 31))
# k_scores = []
# for k in k_range:    
#    knn = KNeighborsClassifier(n_neighbors=k)    
#    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#    k_scores.app

# import matplotlib.pyplot as plt
# %matplotlib inline

# # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


#import data
path = os.path.join(os.getcwd(), "negative_polarity")
os.chdir(path)
data= pd.read_csv('reviews_document_term.csv')
vec=data.drop('class', axis=1)
""" define the parameter values that should be searched """
# logistic regression
C_range = list(range(1, 1000))
nmin_range
min_leafs
ccp_alpha


# specify "parameter distributions" rather than a "parameter grid"
#param_dist = dict(n_neighbors=k_range, weights=weight_options)
param_dist = dict(C=C_range) #, weights=weight_options)
param_dist_dt_r = dict(min_samples_split=nmin_range, min_samples_leaf=min_leafs)


    # min_samples_split=2, 
    # min_samples_leaf=1, 
    # min_weight_fraction_leaf=0.0, 
    # max_features=None, 
# Initialize models
nb=MultinomialNB()
lr=LogisticRegression(penalty='l1', solver='saga', random_state=5)
dt = DecisionTreeClassifier(
    criterion='gini', 
    splitter='best',  
    random_state=5)

rand = RandomizedSearchCV(lr,
                          param_dist,
                          cv=10,
                          scoring='accuracy',
                          n_iter=50,
                          random_state=5,
                          return_train_score=False)
rand.fit(vec, data['class'])
pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

# examine the best model
print(rand.best_score_)
print(rand.best_params_)
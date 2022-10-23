import os
import pandas as pd
import csv
import string
import re
import nltk
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


def read_text(wd, folds, class_label):
    start = True
    for i in folds:  # 6
        # Change the directory
        path = wd + "\\fold" + str(i)
        os.chdir(path)

        # iterate through all file
        for idx, file in enumerate(os.listdir()):
            # Check whether file is in text format or not
            file_path = f"{path}\\{file}"

            if start:
                X = pd.read_csv(file_path, header=None, sep='\s\s+', quoting=csv.QUOTE_NONE, encoding='utf-8',
                                index_col=False, engine="python")
                start = False
            else:
                X = pd.concat([X, pd.read_csv(file_path, header=None, sep='\s\s+', quoting=csv.QUOTE_NONE,
                                              encoding='utf-8', index_col=False, engine="python")])
    X.columns = ['review']

    X['class'] = [class_label] * X.shape[0]
    return X


def preprocess_sentence(data: pd.DataFrame) -> pd.DataFrame:
    # Remove punctuation
    data['review'] = data['review'].apply(lambda text: "".join([i for i in text if i not in string.punctuation]))
    # Remove capital letters
    data['review'] = data['review'].apply(lambda text: text.lower())
    # Remove numbers
    data['review'] = data['review'].apply(lambda text: ''.join([i for i in text if not i.isdigit()]))
    # Tokenize words
    data['review'] = data['review'].apply(lambda text: re.split(r'\s+', text))
    # Lemmatize words
    word_lemmatizer = nltk.stem.WordNetLemmatizer()
    data['review'] = data['review'].apply(
        lambda text: ' '.join([word_lemmatizer.lemmatize(word) for word in text]))  # , [get_wordnet_pos(word)]
    return data


# Download NLP corpora
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Folder Path
wd = os.getcwd()
path = os.path.join(wd, "negative_polarity")
folder_list = ["\\deceptive_from_MTurk", "\\truthful_from_Web"]

# Train: Collect deceptive (0) and truthful (1) reviews
train_data = pd.concat([read_text(wd=path + folder_list[0], folds=range(1, 5), class_label=0),
                        read_text(wd=path + folder_list[1], folds=range(1, 5), class_label=1)])
train_prep = preprocess_sentence(train_data)

# Test set: Collect deceptive (0) and truthful (1) reviews
test_data = pd.concat([read_text(wd=path + folder_list[0], folds=range(5, 6), class_label=0),
                       read_text(wd=path + folder_list[1], folds=range(5, 6), class_label=1)])
test_prep = preprocess_sentence(test_data)

# Vectorize the data excluding stopwords
stop = stopwords.words('english')
ngram_range = (1, 2)
max_features = 1000
use_idf = True

vector = TfidfVectorizer(max_features=max_features,
                         ngram_range=ngram_range,
                         min_df=6, # if they are present in less than the 10% of the sample then not considered
                         max_df=320, # if they appear in more than half of the docoments they are not considered
                         use_idf=use_idf)

train_vec = vector.fit_transform(train_prep['review'])
test_vec = vector.transform(test_prep['review'])

# Train set: Construct document-term matrix
os.chdir(path)
train_df = pd.DataFrame(train_vec.toarray().transpose(), index=vector.get_feature_names())
train_df.columns = ['Rev ' + str(i) for i, _ in enumerate(train_df.columns)]
train_prep=train_prep.reset_index()
train_df=train_df.T
train_df['class']=np.asarray(train_prep['class'])
train_df.to_csv('train_reviews_document_term.csv')

# Test set: Construct document-term matrix 
test_df = pd.DataFrame(test_vec.toarray().transpose(), index=vector.get_feature_names())
test_df.columns = ['Rev ' + str(i) for i, _ in enumerate(test_df.columns)]
test_df=test_df.T
test_df['class']=np.asarray(test_prep['class'])
test_df.to_csv('test_reviews_document_term.csv')


# term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

# # Top 25 words
# term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False)[:25]

# # Print the first 10 rows
# print(term_document_matrix.drop(columns=['total_count']).head(10))
# term_document_matrix['total_count'].plot.bar()

# simulate splitting a dataset of 25 observations into 5 folds
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle = False).split(range(25))

# Import data
path = os.path.join(wd, "negative_polarity")
os.chdir(path)
train_df = pd.read_csv('train_reviews_document_term.csv', header=0, index_col=0)
test_df = pd.read_csv('test_reviews_document_term.csv', header=0, index_col=0)

train_y = train_df['class']
test_y = test_df['class']
train_x = train_df.drop('class', axis=1)
test_x = test_df.drop('class', axis=1)

""" Define the parameter values and distributions"""
# Naive Bayes
smooth = np.arange(0, 1, 0.01)

# Logistic regression
C =np.arange(1, 100, 1)
reg = ['l1', 'l2', 'elasticnet']  # not used

# Decision tree
ccp = np.arange(0, 1, 0.01)
imp = ['gini', 'entropy', 'log_loss']

# Random Forest
m=np.arange(100, 1000, 1)
nfeat = ['sqrt', 'log2']         # sqrt or log2 of total numer of features

param_dist_nb = dict(alpha=smooth)
param_dist_lr = dict(C=C)
param_dist_dt = dict(ccp_alpha=ccp)#, criterion=imp)
param_dist_dt2 = dict(min_samples_split=np.arange(2,10,1), min_samples_leaf=np.arange(1,10,1)) # using min_leaf and n_min
param_dist_rf = dict(ccp_alpha=ccp, max_features=nfeat, n_estimators=m)   #, criterion=imp)

# Initialize models
nb = MultinomialNB()
lr = LogisticRegression(penalty='l1',
                        solver='saga',
                        random_state=5,
                        n_jobs=os.cpu_count(),
                        max_iter=300)
dt = DecisionTreeClassifier(random_state=5)  #,n_jobs=os.cpu_count())
rf = RandomForestClassifier(criterion='gini',
                            random_state=5, n_jobs=os.cpu_count())
models = [nb,  dt, rf] #[lr,
param_dists = [param_dist_nb, param_dist_dt2, param_dist_rf] # param_dist_lr,
model_names=['NB','DT','RF'] # 'LR',


# Train anf fit the model with best parameters.
# Test and print the required measures of performances
selection = []
for i, _ in enumerate(models):
    rand = RandomizedSearchCV(models[i],
                              param_dists[i],
                              cv=5,  # 20
                              scoring='f1', # 'accuracy'; using both gives problems with refit (need to chose according to which score to refit)
                              n_iter=10,  # 100
                              random_state=5,
                              return_train_score=False,
                              verbose=1,
                              refit=True)

    rand.fit(train_x, train_y)
    pred_y=rand.predict(test_x)
    prob_y= rand.predict_proba(test_x)
    print(f'best estimator: {rand.best_estimator_}, score of best estimator: {rand.best_score_}, best parameters setting: {rand.best_params_} ')
    print(f'model: {model_names[i]}, accuracy: {accuracy_score(test_y,pred_y)}, precision: {precision_score(test_y,pred_y)}, recall: {recall_score(test_y, pred_y)}, F1 : {f1_score(test_y,pred_y)}')
    
    # d=pd.DataFrame(rand.cv_results_)
    # pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    # selection.append(f'{models[i]}: Best parameters {rand.best_score_}, best score  {rand.best_score_}')

# print(selection)

print()

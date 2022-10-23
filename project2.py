import os
import pandas as pd
import csv
import string
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


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
                         min_df=1,
                         max_df=1.0,
                         use_idf=use_idf)

train_vec = vector.fit_transform(train_prep['review'])
test_vec = vector.transform(test_prep['review'])

# Train set: Construct document-term matrix
os.chdir(path)
train_df = pd.DataFrame(train_vec.toarray().transpose(), index=vector.get_feature_names())
train_df.columns = ['Rev ' + str(i) for i, _ in enumerate(train_df.columns)]
train_df = pd.concat([train_df.T, train_prep['class']], axis=1)
train_df.to_csv('train_reviews_document_term.csv')

# Test set: Construct document-term matrix 
test_df = pd.DataFrame(test_vec.toarray().transpose(), index=vector.get_feature_names())
test_df.columns = ['Rev ' + str(i) for i, _ in enumerate(test_df.columns)]
test_df = pd.concat([test_df.T, test_prep['class']], axis=1)
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
smooth = list(range(start=0, stop=1, step=0.01))

# Logistic regression
C = list(range(start=1, stop=100, step=1))
reg = ['l1', 'l2', 'elasticnet']

# Decision tree
ccp = list(range(start=0, stop=1, step=0.01))
imp = ['gini', 'entropy', 'log_loss']

# Random Forest
m = list(range(start=100, stop=1000, step=1))
nfeat = list

param_dist_nb = dict(alpha=smooth)
param_dist_lr = dict(C=C)
param_dist_dt = dict(ccp_alpha=ccp, criterion=imp)
param_dist_rf = dict(ccp_alpha=ccp, max_features=nfeat, n_estimators=m, criterion=imp)

# Initialize models
nb = MultinomialNB()
lr = LogisticRegression(solver='saga',
                        random_state=5,
                        n_jobs=os.cpu_count(),
                        max_iterint=250)
dt = DecisionTreeClassifier(random_state=5,
                            n_jobs=os.cpu_count())
rf = RandomForestClassifier(criterion='gini',
                            random_state=5,
                            n_jobs=os.cpu_count())
models = [nb, lr, dt, rf]
param_dists = [param_dist_nb, param_dist_lr, param_dist_dt, param_dist_rf]

# Train
selection = []
for i, _ in enumerate(models):
    rand = RandomizedSearchCV(models[i],
                              param_dists[i],
                              cv=2,  # 20
                              scoring=['accuracy', 'f1'],
                              n_iter=1,  # 100
                              random_state=5,
                              return_train_score=False,
                              verbose=1)
    rand.fit(train_x, train_y)
    pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

    selection.append(f'{models[i]}: Best parameters {rand.best_score_}, best score  {rand.best_score_}')

selection

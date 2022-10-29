import os
import pandas as pd
import csv
import string
import re
import nltk
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_selection import mutual_info_classif


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

vector = TfidfVectorizer(ngram_range=ngram_range,
                         min_df=6,  # if they are present in less than the 10% of the sample then not considered
                         max_df=320,  # if they appear in more than half of the documents they are not considered
                         use_idf=use_idf)

train_vec = vector.fit_transform(train_prep['review'])
test_vec = vector.transform(test_prep['review'])

# Train set: Construct document-term matrix
os.chdir(path)
train_df = pd.DataFrame(train_vec.toarray().transpose(), index=vector.get_feature_names_out())
train_df.columns = ['Rev ' + str(i) for i, _ in enumerate(train_df.columns)]
train_prep = train_prep.reset_index()
train_df = train_df.T
train_df['class'] = np.asarray(train_prep['class'])
train_df.to_csv('train_reviews_document_term.csv')

# Test set: Construct document-term matrix 
test_df = pd.DataFrame(test_vec.toarray().transpose(), index=vector.get_feature_names_out())
test_df.columns = ['Rev ' + str(i) for i, _ in enumerate(test_df.columns)]
test_df = test_df.T
test_df['class'] = np.asarray(test_prep['class'])
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

# Mutual information criterios for feature selection
mic = mutual_info_classif(train_x, train_y, random_state=5)

# Training set. Feature selection with Mutal information Criterion
# Exclude features that are independent of the class and sort the trainings data based on the mic score
train_x_mic = train_x.loc[:, mic > 0.01]
train_x_mic.loc['mic'] = mic[mic > 0.01]
train_x_mic.sort_values(by=['mic'], inplace=True, axis=1, ascending=False)
train_x = train_x_mic.drop('mic', axis=0)

# Test set. Feature selection with Mutal information Criterion
# Exclude features that are independent of the class and sort the trainings data based on the mic score
test_x_mic = test_x.loc[:, mic > 0.01]
test_x_mic.loc['mic'] = mic[mic > 0.01]
test_x_mic.sort_values(by=['mic'], inplace=True, axis=1, ascending=False)
test_x = test_x_mic.drop('mic', axis=0)

"""Histograms"""
# histogram of mic
plt.hist(mic)
#plt.show()

# histogram of train dataset, without considerign too frequent and too infrequent words, as well as the less informative ones
#
# t = train_x.replace(0, np.nan)
# t.plot.hist()
# plt.show()

""" Define the parameter values and distributions"""
# Naive Bayes
smooth = np.arange(0.01, 1, 0.01)
n_iter = 50
micefeat_range = range(250, train_x.shape[1] - 1)

# Logistic regression
C = np.arange(1, 300, 1)

# Decision tree
ccp = np.arange(0, 1, 0.01)
imp = ['gini', 'entropy', 'log_loss']

# Random Forest
m = np.arange(100, 500, 1)
nfeat = np.arange(1, int((train_x.shape[1]) / 2), 1)

param_dist_nb = dict(alpha=smooth)
param_dist_lr = dict(C=C)
param_dist_dt = dict(ccp_alpha=ccp)  # , criterion=imp)
param_dist_dt2 = dict(min_samples_split=np.arange(2, 10, 1),
                      min_samples_leaf=np.arange(1, 10, 1))  # using min_leaf and n_min
param_dist_rf = dict(ccp_alpha=ccp,
                     max_features=nfeat,
                     n_estimators=m)  # , criterion=imp)

# Initialize models
nb = MultinomialNB()
lr = LogisticRegression(penalty='l1',
                        solver='liblinear',
                        random_state=5,
                        max_iter=300)
dt = DecisionTreeClassifier(random_state=5)  # ,n_jobs=os.cpu_count())
rf = RandomForestClassifier(criterion='gini',
                            random_state=5,
                            n_jobs=os.cpu_count())
models = [dt, dt, rf, lr]
param_dists = [param_dist_dt, param_dist_dt2, param_dist_rf, param_dist_lr]
model_names = ['DT_rule', 'DT_ccp', 'RF', 'LR']

# # Train anf fit the model with best parameters.
# selection = []
# model_predictions=[]
# for i, _ in enumerate(models):

#     rand = RandomizedSearchCV(models[i],
#                               param_dists[i],
#                               cv=5,  # 20
#                               scoring='f1', 
#                               n_iter=50,  # 200
#                               random_state=5,
#                               return_train_score=False,
#                               verbose=1,
#                               refit=True)

#     rand.fit(train_x, train_y)

# # Test and print the required measures of performances
#     pred_y=rand.predict(test_x)    
#     model_predictions.append(pred_y)
#     # print(f'best estimator: {rand.best_estimator_}, score of best estimator: {rand.best_score_}, best parameters setting: {rand.best_params_} ')
#     print(f'model: {model_names[i]}, accuracy: {accuracy_score(test_y,pred_y)}, precision: {precision_score(test_y,pred_y)}, recall: {recall_score(test_y, pred_y)}, F1 : {f1_score(test_y,pred_y)}')
#     print(confusion_matrix(test_y, pred_y))
#     # d=pd.DataFrame(rand.cv_results_)
#     # pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
#     # selection.append(f'{models[i]}: Best parameters {rand.best_score_}, best score  {rand.best_score_}')

model_names.append('NB')
micfeat = random.choices(micefeat_range, k=n_iter)
best_nb = None
score = []
params = []
for i in range(n_iter):
    train_x_nb = train_x.iloc[:, 0:micfeat[i]]
    rand = RandomizedSearchCV(nb,
                              param_dist_nb,
                              cv=5,  # 20
                              scoring='f1',
                              n_iter=1,  # 200
                              random_state=5,
                              return_train_score=True,
                              verbose=0,
                              refit=True)

    rand.fit(train_x_nb, train_y)
    score.append(rand.best_score_)
    params.append([rand.best_params_, micfeat[i]])

best_param = params[score.index(max(score))]
train_x_best_nb = train_x.iloc[:, 0:best_param[1]]

best_nb = MultinomialNB(alpha=best_param[0]['alpha'])
best_nb.fit(train_x_best_nb, train_y)

model_predictions.append(pred_y)

print(f'model: {model_names[-1]},'
      f' accuracy: {accuracy_score(test_y, pred_y)},'
      f' precision: {precision_score(test_y, pred_y)},'
      f' recall: {recall_score(test_y, pred_y)},'
      f' F1 : {f1_score(test_y, pred_y)}')
print(confusion_matrix(test_y, pred_y))

for i in range(len(model_predictions)):
    for j in range(len(model_predictions)):
        if i < j:
            table = pd.crosstab(model_predictions[i], model_predictions[j])
            print(f"cross table of {model_names[i]} with {model_names[j]}: {table}")
            result = mcnemar(table, exact=True)
            print(f"MCNEMAR TEST--> statistic value: {result.statistic}, pvalue: {result.pvalue} ")



# TODO include everything with mic>0.01, make histogram to show why it's a good value
# TODO include a histogram of all words, without infrequent terms and stop words and without mic<0.1
# TODO show the top 5 most relevant words for each class seperatly
# TODO add the statistical test for the models (whether they are significantly better)
# TODO crosstable
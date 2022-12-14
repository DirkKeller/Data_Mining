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
from skopt import BayesSearchCV
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
    data['review'] = data['review'].apply(lambda text: ' '.join([word_lemmatizer.lemmatize(word) for word in text]))  # , [get_wordnet_pos(word)]
    
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
ngram_range = (1, 1)
use_idf = True

vector = TfidfVectorizer(ngram_range=ngram_range,
                         min_df=6,  # if they are present in less than the 10% of the sample then not considered
                         max_df=320,  # if they appear in more than half of the documents they are not considered
                         use_idf=use_idf,
                         stop_words=stop)

train_vec = vector.fit_transform(train_prep['review'])
test_vec = vector.transform(test_prep['review'])

# Train set: Construct document-term matrix
os.chdir(path)
train_df = pd.DataFrame(train_vec.toarray().transpose(), index=vector.get_feature_names())
train_df.columns = ['Rev ' + str(i) for i, _ in enumerate(train_df.columns)]
train_prep = train_prep.reset_index()
train_df = train_df.T
train_df.columns= vector.get_feature_names()
train_df['class'] = np.asarray(train_prep['class'])
train_df.to_csv('train_reviews_document_term.csv')

# Test set: Construct document-term matrix 
test_df = pd.DataFrame(test_vec.toarray().transpose(), index=vector.get_feature_names())
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
train_x_truth=train_x.iloc[:320,:]
train_x_fake=train_x.iloc[320:,:]



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
# plt.hist(mic)
# plt.show()

"""Most important features according to mic"""
# #train_x_mic_truth è il dataset di quelli classificati come truth, sorted per mic, quindi basta prendere le prime 5 colonne
# train_x_mic_truth=train_x_truth.loc[:,mic > 0.01]
# train_x_mic_truth.loc['mic'] = mic[mic> 0.01]
# train_x_mic_fake=train_x_fake.loc[:,mic> 0.01]
# train_x_mic_fake.loc['mic'] = mic[mic > 0.01]
# train_x_mic_truth.sort_values(by=['mic'], inplace=True, axis=1, ascending=False)
# train_x_mic_fake.sort_values(by=['mic'], inplace=True, axis=1, ascending=False)
# top5_words_truth=train_x_mic_truth.columns[:5]
# top5_words_fake=train_x_mic_fake.columns[-5:]
# print(f"Top 5 words truth: {top5_words_truth} with MIC values of: {train_x_mic_truth.iloc[-1,:5]}")
# print(f"Top 5 words fake: {top5_words_fake} with MIC values of: {train_x_mic_fake.iloc[-1,-5:]}")



""" Define the parameter values and distributions"""
# Naive Bayes
smooth = np.arange(0.01, 1, 0.001)
n_iter = 50
micefeat_range = range(150, train_x.shape[1] - 1)

# Logistic regression
C = np.arange(1, 1000, 1)

# Decision tree
ccp = np.arange(0, 1, 0.001)
nmin=np.arange(2, 20, 1)
minleaf=np.arange(1, 20, 1)

# Random Forest
m = np.arange(50, 200, 1)
nfeat = np.arange(1, int((train_x.shape[1])/2), 1)

param_dist_nb = dict(alpha=smooth)
param_dist_lr = dict(C=C)
param_dist_dt = dict(ccp_alpha=ccp)  # , criterion=imp)
param_dist_dt2 = dict(min_samples_split=nmin,
                      min_samples_leaf=minleaf)  # using min_leaf and n_min
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
model_names = ['DT_ccp', 'DT_rule', 'RF', 'LR']

# # Train anf fit the model with best parameters.
selection = []
model_predictions=[]
best_models=[]
for i, _ in enumerate(models):
    print(f'-------------------------------------------------------------------\n',
          f'------------------------ {model_names[i]} ----------------------------\n',
          f'-------------------------------------------------------------------\n')

    model_bay = BayesSearchCV(models[i],
                              param_dists[i],
                              cv=30,  # 20
                              scoring='roc_auc', 
                              n_iter=50,  # 200
                              random_state=5,
                              return_train_score=False,
                              verbose=1,
                              refit=True,
                              n_jobs=os.cpu_count())

    model_bay.fit(train_x, train_y)
    best_models.append(model_bay.best_estimator_)
    print(f'best estimator: {model_bay.best_estimator_}, score of best estimator: {model_bay.best_score_}, best parameters setting: {model_bay.best_params_} ')
    
    # Test and print the required measures of performances
    model_full_train = model_bay.best_estimator_.fit(train_x, train_y)  
    pred_y=model_full_train.predict(test_x)    
    model_predictions.append(pred_y)
    print(f'model: {model_names[i]}, accuracy: {accuracy_score(test_y,pred_y)}, precision: {precision_score(test_y,pred_y)}, recall: {recall_score(test_y, pred_y)}, F1 : {f1_score(test_y,pred_y)}')
    print(confusion_matrix(test_y, pred_y))

    
    d=pd.DataFrame(model_bay.cv_results_)
    pd.DataFrame(model_bay.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    selection.append(f'{models[i]}: Best parameters {model_bay.best_score_}, best score  {model_bay.best_score_}')

    """for j, param in enumerate(param_dists[i]):
        bay = model_bay.cv_results_['param_'+param]

        fig = plt.figure(figsize=(15, 7))

        ax = plt.gca()
        ax.scatter(np.arange(len(bay)),
                   bay,
                   color=[(random.random(), random.random(), random.random())],
                   s=20,
                   label=model_names[i] + ': param value');
        ax.scatter(np.arange(len(bay)),
                   model_bay.cv_results_['mean_test_score'],
                   color=[(random.random(), random.random(), random.random())],
                   s=20,
                   label=model_names[i] + ': test score');

        #ax.set_yscale('log')
        # prev: bay = model_bay.cv_results_['param_'+param]
        # scatter(np.arange(len(bay)), bay)
        plt.legend();
        plt.title(param);
        #os.chdir('/content')
        plt.show()"""


#Get the five terms that most support classification
#for DecisionTreeClassifiers use ".feature_importances_"
#for RandomForestClassifiers use ".feature_importances_"
#for LogisticRegression use ".coef_" 
#for MultinomialNB use ".feature_log_prob_" 

#not sure what to do next
#p_dt1 = best_models[0].feature_importances_
#p_dt2 = best_models[1].feature_importances_
#p_rf = best_models[2].feature_importances_
#p_lr = best_models[3].coef_


model_names.append('NB')
micfeat = random.choices(micefeat_range, k=n_iter)
best_nb = None
score = []
params = []
for i in range(n_iter):
    train_x_nb = train_x.iloc[:, 0:micfeat[i]]
    bay = BayesSearchCV(nb,
                        param_dist_nb,
                        cv=20,
                        scoring='roc_auc',
                        n_iter=50,  
                        random_state=5,
                        return_train_score=False,
                        verbose=1,
                        refit=True ,
                        n_jobs=os.cpu_count())

    bay.fit(train_x_nb, train_y)
    score.append(bay.best_score_)
    params.append([bay.best_params_, micfeat[i]])

################################################
"""bay_nb = bay.cv_results_['param_alpha']
fig = plt.figure(figsize=(15, 7))

ax = plt.gca()
ax.scatter(np.arange(len(bay_nb)),
            bay_nb,
            color=[(random.random(), random.random(), random.random())],
            s=20,
            label='Naive Bayes : param value');
ax.scatter(np.arange(len(bay_nb)),
            bay.cv_results_['mean_test_score'],
            color=[(random.random(), random.random(), random.random())],
            s=20,
            label= 'Naive Bayes : test score');

#ax.set_yscale('log')
# prev: bay = model_bay.cv_results_['param_'+param]
# scatter(np.arange(len(bay)), bay)
plt.legend();
plt.title('alpha');
#os.chdir('/content')
plt.show()
###############################################################"""

best_param = params[score.index(max(score))]
train_x_best_nb = train_x.iloc[:, 0:best_param[1]]
test_x_best_nb = test_x.iloc[:, 0:best_param[1]]

best_nb = MultinomialNB(alpha=best_param[0]['alpha'])
best_models.append(best_nb)
best_nb.fit(train_x_best_nb, train_y)
pred_y = best_nb.predict(test_x_best_nb)
model_predictions.append(pred_y)


# top 5 terms supporting truthful/deceptive
p = best_nb.feature_log_prob_
prob_truth = p[0,:]
prob_fake = p[1,:]
ratio = prob_truth-prob_fake # log odds
top5_index_truth = np.flip(np.argsort(ratio))[:5] # supporting truthful 
top5_index_fake = np.flip(np.argsort(ratio))[-5:] # supporting deceptive
top5_truth = test_x_best_nb.columns[top5_index_truth]
top5_fake= test_x_best_nb.columns[top5_index_fake]
print(f"Top 5 supporting truthful: {top5_truth}")
print(f"Top 5 supporting deceptive: {top5_fake}")
# index_top5_truth = np.flip(np.argsort(prob_truth))[:5]
# index_top5_fake = np.flip(np.argsort(prob_fake))[:5]
# top5_terms_truth = test_x_best_nb.columns[index_top5_truth]
# top5_terms_fake = test_x_best_nb.columns[index_top5_fake]
# print(f"The five terms that most support a truthful review for NB are: {top5_terms_truth}")
# print(f"The five terms that most support a deceptive review for NB are: {top5_terms_fake}")




print(f'model: {model_names[-1]},'
      f' accuracy: {accuracy_score(test_y, pred_y)},'
      f' precision: {precision_score(test_y, pred_y)},'
      f' recall: {recall_score(test_y, pred_y)},'
      f' F1 : {f1_score(test_y, pred_y)}')
a = best_param[0]['alpha']
print(f'Best model params: alpha = {a},'
      f'number of features = {best_param[1]}')
#     
print(confusion_matrix(test_y, pred_y))

for i in range(len(model_predictions)):
    bool_i=model_predictions[i]==test_y
    for j in range(len(model_predictions)):
        if i < j:
          bool_j= model_predictions[j]==test_y
          table = pd.crosstab(bool_i, bool_j)
          print(f"cross table of {model_names[i]} with {model_names[j]}: {table}")
          result = mcnemar(table, exact=True)
          print(f"MCNEMAR TEST--> statistics value: {result.statistic}, pvalue: {result.pvalue} ")



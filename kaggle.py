# Importing Necessary libraries for text cleaning and performance tuning

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, validation_curve

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

# importing 10,000 movie reviews that were extracted from IMDB into a pandas dataframe
 
kaggle_df = pd.read_csv('train.csv')

# Dataframe has two column labels: "Text" and "Label" 
#Checking for dataframe column lables
# Notice that the reviews are pre-lableld as either (1 - positive) or (0 - negative)
kaggle_df.keys()

#Splitting the data frame columns into 2 separate variable lists
x = kaggle_df['text']
y= kaggle_df['label']

# Splitting the variables into training (75%) and testing (25%) sets with complete random states

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Converting training set to tokens of common vocabulary words
#extracting elements from our text column
count_vector = CountVectorizer()
X_train_counted = count_vector.fit_transform(X_train)
print(X_train_counted.shape)

# calculate the occurence of word tokens using Tfidftransformer and assign weights (Tf-idf scores) based on frequency.
# We also calculate rarity of word in document corpus using Inverse Document Frequency

document_transformer = TfidfTransformer()
X_trained_transformer = document_transformer.fit_transform(X_train_counted)
X_trained_transformer.shape

# Classifying word occurence using Naive Bayes classifier to further train our data
nbc = MultinomialNB().fit(X_trained_transformer, y_train)


#predicting outcome of test set
X_test_new_counts = count_vector.transform(x)
X_test_new_tfidf = document_transformer.transform(X_test_new_counts)

predicted1= nbc.predict(X_test_new_tfidf)

# <<OPTIONAL STEP>>
# Building pipeline to combine the 3 previous steps

text_nbc = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_nbc = text_nbc.fit(X_train,y_train)

# <<END OPTIONAL STEP/>>


# lets first manually clean our testing set
file_2 = pd.read_csv('test.csv')
#file_2
x_tester=file_2['text']
y_tester=file_2['Id']
x_tester_cleaned = []
for x in x_tester:
    sentence=x
    pattern = re.compile('[^\w\s]')
    pattern.findall(x)
    new_sentence2 = pattern.sub("",x)
    print(new_sentence)
    x_tester_cleaned.append(new_sentence2)
import csv
count = 0
with open('test.csv','w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id','text'])
    for w, x in zip(y_tester, x_tester_cleaned):
        writer.writerow([count,x])
        count = count+1

# testing set     
x_test = file_2['text']
x_test

#Now let us test our models on our testing set (this prediction is a lot better than the previous one)
docs_test = x_test
predicted2 = text_nbc.predict(docs_test)
np.mean(predicted2)

# Importing cleaned data
import csv
count = 0
with open('sub2.csv','w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id','Category'])
    for w, x in zip(x_test, predicted2):
        writer.writerow([count,x])
        count = count+1


# Now let us test the performance of the classifier
# kaggle_test = kaggle_df(subset='test', shuffle=True)
# 86.4% predictive accuracy

predicted = text_nbc.predict(X_test)
np.mean(predicted == y_test)

# with less noisy data, let's training support vector machines and testing performance
# 87.52% predictive accuracy
from sklearn.linear_model import SGDClassifier
text_clf_svm2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=4500, random_state=200, tol=None))])

text_clf_svm2 = text_clf_svm2.fit(x,y)
predicted_svm = text_clf_svm2.predict(x_test)
np.mean(predicted_svm)


# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}


# NLTK
# Removing stop words to improve prediction
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])

import nltk
nltk.download('stopwords')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(x_train,y_train)

predicted_mnb_stemmed = text_mnb_stemmed.predict(x_test)

np.mean(predicted_mnb_stemmed)


# Next, we create an instance of the grid search by passing the classifier, parameters 
# and n_jobs=-1 which tells to use multiple cores from user machine.

gs_clf = GridSearchCV(text_nbc, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)

# To see the best mean score and the params, run the following code

gs_clf.best_score_
#gs_clf.best_params_

# Output for above should be: The accuracy has now increased to ~90.6% for the NB classifier (not so naive anymore! 😄)
# and the corresponding parameters are {‘clf__alpha’: 0.01, ‘tfidf__use_idf’: True, ‘vect__ngram_range’: (1, 2)}.

# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)


gs_clf_svm.best_score_
#gs_clf_svm.best_params_
import numpy as np
import re
import matplotlib.pyplot as plt
#import scikitplot as skplt

#import NLTK methods
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

#Import processing methods
from sklearn import preprocessing, metrics, cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import load_files, make_classification,make_blobs,make_gaussian_quantiles, load_digits
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict

#Import Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB


def data_cleaner(l):
    """
    :param l: line to be cleaned.
    :return line: cleaned line.

    Removes specified elements of a particular line from the dataset.
    """

    line = re.sub(r'\W', ' ', l)  # remove all the special characters
    line = re.sub(r'\s+[a-zA-Z]\s+', ' ', line)  # remove all single characters
    line = re.sub(r'\^[a-zA-Z]\s+', ' ', line)  # Remove single characters from the start
    line = re.sub(r'\s+', ' ', line, flags=re.I)  # Substituting multiple spaces with single space
    line = re.sub(r'^b\s+', '', line)  # Removing prefixed 'b'
    line = re.sub(" \d+", " ", line)  # Remove digits
    line = line.replace('_', '')  # Remove the _ in the document and replace them with ''
    line = line.lower()  # Convert to lowercase text
    return line


def lemmatise(line, lmtzr):
    """
    :param line: line to be lemmatised.
    :param lmtzr: lemmatiser object.
    :return: joined lemmatised tokens as one list.

    Tokenize the text, reduce the tokens to their lemmas, return the joined tokens as one list

    """
    list = []
    words = nltk.word_tokenize(line)
    for item in words:
        item = lmtzr.lemmatize(item)
        list.append(item)
    return ' '.join(list)


def preprocessing(path):
    """
    :param path: path of the test / train data.
    :return: preprocessed_data

    Clean and lemmatize the dataset.

    """

    spam_data = load_files(path)
    preprocessed_data = []

    lmtzr = WordNetLemmatizer()  # Instantiates new lemmatiser object.
    X, y = spam_data.data, spam_data.target
    print('Lengths of X and Y parameters.', len(X), len(y))

    #for i in range(0, 30):
        #print(y[i], X[i])
    #exit()
    #for each in y:
        #print(each)
    #exit()

    print('Cleaning and Lemmatising the data.')
    for each in range(0, len(X)):
        document = data_cleaner(str(X[each])) #Data Cleaning
        document = lemmatise(document, lmtzr) #Lemmatise
        preprocessed_data.append(document)
    print('Data is preprocessed.')
    print('____________________________________________________________________________________')
    return preprocessed_data, X, y


def prepare_for_fitness(preprocessed_data):
    """
    :return: X, count_vect, tdidf_transformer

    Prepare the preprocessed_data into the model. Convert 'text' into 'numbers'.

    1. Vectorise
    2. TD IDF

    """

    print('Vectorising.')
    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    X = count_vect.fit_transform(preprocessed_data)
    print('Vectorised.')
    print('TD IDF.')
    tfidf_transformer = TfidfTransformer()
    print('TD IDF done.')
    print('Fitting the model.')
    print('____________________________________________________________________________________')
    print('\n')
    X = tfidf_transformer.fit_transform(X)

    return X, count_vect, tfidf_transformer


def test_classifier(classifier_no, X1, y, count_vect, tfidf_transformer):
    """
    :param classifier_no: 0-1-2
    :param documents:
    :param X1:
    :param y:
    :param count_vect:
    :param tfidf_transformer:
    :return: accuracy score. The accuracy score of a classifier can be computed as
            (TP + TN) / (TP + FP + FN + TN)

    1. Fit Classifier
    2. Load and Prepare Test Data
    3. Test data in the Fitted Classifier
    4. Report Objective Metrics

    """
    if classifier_no == 0:
        text_clf = MultinomialNB().fit(X1, y)
    if classifier_no == 1:
        text_clf = SGDClassifier()
        text_clf = SGDClassifier().fit(X1, y)
    if classifier_no == 2:
        text_clf = RandomForestClassifier()
        text_clf = RandomForestClassifier().fit(X1, y)

    print('Results for', text_clf.__class__)
    test_path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/test-mails'
    test_data = load_files(test_path)
    X_test, y_test = test_data.data, test_data.target

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = text_clf.predict(X_test_tfidf)

    print('The accuracy score: ', np.mean(predicted == y_test))
    print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
    print('confusion matrix', '\n', metrics.confusion_matrix(test_data.target, predicted))
    print('____________________________________________________________________________________')

    #mat = confusion_matrix(test_data.target, labels)
    #sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
    #xticklabels=train.target_names, yticklabels=train.target_names)
    #plt.xlabel('true label')
    #plt.ylabel('predicted label');

    # predictions = cross_val_predict(text_clf, X,y)
    # skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
    # plt.show()

    return np.mean(predicted == y_test)


"""
Start the program with the dataset parameter p.
p = 0 // Subset of Ling Spam Dataset.
p = 1 // Whole enron Dataset.
p = 2 // Own Dataset. 
"""

p = 0
print('Classification Started')

if p == 0:
    path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/train-mails'
    pd, X, y = preprocessing(path)
    X1, v, t = prepare_for_fitness(pd)
    a = test_classifier(0, X1, y, v, t)
    b = test_classifier(1, X1, y, v, t)
    c = test_classifier(2, X1, y, v, t)
else:
    path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/enron/train-data'
    pd, X, y = preprocessing(path)
    X1, v, t = prepare_for_fitness(pd)
    a = test_classifier(0, X1, y, v, t)
    b = test_classifier(1, X1, y, v, t)
    c = test_classifier(2, X1, y, v, t)

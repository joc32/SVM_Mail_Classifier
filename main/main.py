#Import general libraries
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#import NLTK methods
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

#Import processing methods
from sklearn import preprocessing, metrics, cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import load_files, make_classification, make_blobs, make_gaussian_quantiles, load_digits
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict

#Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions


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
    bad_words = ['nthe','ha','nto','wa','hou','na','nand','nenron','ncc','nfrom','nthis','nplease','nthanks','ni','nsubject','ou','we']
    for item in words:
        item = lmtzr.lemmatize(item)
        if item in bad_words:
            item = item[1:]
        if item == 'ou':
            item = item[1:]
        if item == 'ect':
            item = ''
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
    print('Cleaning and Lemmatising the data.')

    for each in range(0, len(X)):
        document = data_cleaner(str(X[each])) #Data Cleaning
        document = lemmatise(document, lmtzr) #Lemmatise
        preprocessed_data.append(document)

    print('Data is preprocessed.')
    print('____________________________________________________________________________________')
    return preprocessed_data, X, y


def plot_top_frequencies(x, y, total_sum):
    """
    :param words_freq:
    :return:

    Plots the frequency of the top K words in the built dictionary.

    """

    plt.bar(x, y, width=0.6)
    s = str(total_sum)
    plt.title('Frequency Count of the top 10 words. Σ of all Counts: %i' % total_sum)
    plt.ylabel('Frequency in %')
    plt.xlabel('Word')
    plt.show()

def prepare_for_fitness(preprocessed_data):
    """
    :param preprocessed_data: preprocessed data
    :return: X, count_vect, tdidf_transformer

    Prepare the preprocessed_data into the model. Convert 'text' into 'numbers'.

    1. Vectorise
    2. TD IDF

    """
    print('Vectorising.')

    count_vect = CountVectorizer(max_features=1000, stop_words=stopwords.words('english'))
    X = count_vect.fit_transform(preprocessed_data)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in count_vect.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_sum = int(sum_words.sum(axis=1))
    x, y = [], []
    k_words = 0

    # can be changed easily to counts, remove the / total sum
    for k, v in words_freq:
        if k_words == 10:
            break
        else:
            x.append(k)
            y.append((v/total_sum)*100)
            k_words += 1

    plot_top_frequencies(x, y, total_sum)
    # print((words_freq)) #print the dictionary

    print('Vectorised.')
    print('TD IDF.')
    tfidf_transformer = TfidfTransformer()
    print('TD IDF done.')
    print('Fitting the model.')
    print('____________________________________________________________________________________')
    print('\n')
    X = tfidf_transformer.fit_transform(X)

    return X, count_vect, tfidf_transformer


def plot_matrix(mat, test_data, classifier_name):
    """
    :param mat: Confusion matrix to be plotted.
    :param test_data: Test_data
    :param classifier_name: Classifier Name
    :return: none

    Plots the confusion matrix for a given classifier using Seaborn's heatmap module.
    """

    title = 'Confusion Matrix for ' + classifier_name
    sns.set(font_scale=1.4)
    sns.heatmap(mat, square=True, annot=True, annot_kws={"size": 30}, fmt='d', cbar=False,
                xticklabels=test_data.target_names, yticklabels=test_data.target_names)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title(title)
    plt.show()


def test_classifier(classifier_no, X1, y, count_vect, tfidf_transformer):
    """
    :param classifier_no: 0-1-2
    :param documents:
    :param X1: Sparse Matrix
    :param y: Target Names
    :param count_vect: Vectoriser
    :param tfidf_transformer: Transformer
    :return: accuracy score. The accuracy score of a classifier can be computed as
            (TP + TN) / (TP + FP + FN + TN)

    1. Fit Classifier
    2. Load and Prepare Test Data
    3. Test data in the Fitted Classifier
    4. Report Objective Metrics

    """
    if classifier_no == 0:
        text_clf = MultinomialNB().fit(X1, y)
        text_clf.fit(X1, y)
    if classifier_no == 1:
        text_clf = svm.SVC(kernel='linear', C=1.0)
        text_clf.fit(X1, y)
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

    mat = confusion_matrix(test_data.target, predicted)
    plot_matrix(mat, test_data, str(text_clf.__class__))

    return np.mean(predicted == y_test)


"""
Start the program with the dataset parameter param.
param = lingspam // Subset of Ling Spam Dataset.
param = enron // Whole enron Dataset.
param = own // Own Dataset. 
"""


param = input("\n ENTER FILE MODE: "
                  "\n <lingspam> for ling spam data set"
                  "\n <enron> for enron dataset"
                  "\n <own> for own dataset "
                  "\n")
param = 'lingspam'
if param == 'lingspam':
    print('Classification Started')
    print('Classifiers running on Ling Spam dataset subset')
    path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/train-mails'
    pd, X, y = preprocessing(path)
    X1, v, t = prepare_for_fitness(pd)
    accuracy = test_classifier(0, X1, y, v, t)
    accuracy = test_classifier(1, X1, y, v, t)
    accuracy = test_classifier(2, X1, y, v, t)
elif param == 'enron':
    print('Classification Started')
    print('Classifiers running on Enron subset')
    path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/enron/train-data'
    pd, X, y = preprocessing(path)
    X1, v, t = prepare_for_fitness(pd)
    accuracy = test_classifier(0, X1, y, v, t)
    accuracy = test_classifier(1, X1, y, v, t)
    accuracy = test_classifier(2, X1, y, v, t)
elif param == 'own':
    print('\n')
    user_path = input("Please specify the absolute path of your training dataset: \n"
                      "The path has be in the following format: \n"
                      "\n"
                      "../path to your training set/"
                      "\n    |"
                      "\n     -- <subdirectory category1>"
                      "\n    |"
                      "\n     -- <subdirectory category2>"
                      "\n \nThe subdirectories have to include data that already belongs to either category"
                      "\nWhen ready press ENTER key or press CTRL-C to stop the process.\n")
    print('Classification Started. Training from', user_path)
    pd, X, y = preprocessing(user_path)
    X1, v, t = prepare_for_fitness(pd)
    accuracy = test_classifier(0, X1, y, v, t)
    accuracy = test_classifier(1, X1, y, v, t)
    accuracy = test_classifier(2, X1, y, v, t)
else:
    print('Invalid dataset parameter. Killing program.')

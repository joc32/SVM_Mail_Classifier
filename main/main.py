#Import general libraries
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Import NLTK methods
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

    Removes specified elements of a particular line from a dataset using regular expressions.
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
    bad_words = ['nthe', 'ha', 'nto', 'wa', 'hou', 'na', 'nand', 'nenron', 'ncc', 'nfrom', 'nthis', 'nplease', 'nthanks', 'ni', 'nsubject', 'ou', 'we']
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

    Preprocess the dataset by applying the cleaning and lemmatising function on every file.
    """
    train_data = load_files(path)
    preprocessed_data = []
    lmtzr = WordNetLemmatizer()
    X, y = train_data.data, train_data.target


    print('Lengths of X and Y parameters.', len(X), len(y))
    print('Cleaning and Lemmatising the data.')

    for each in range(0, len(X)):
        document = data_cleaner(str(X[each]))
        document = lemmatise(document, lmtzr)
        preprocessed_data.append(document)

    print('Data is preprocessed.')
    print('____________________________________________________________________________________')
    return preprocessed_data, X, y


def return_dictionary_stats(X, vocabulary_items, k_words, param):
    """
    :param words_freq:

    Either plots the frequency of the top K words in the newly built dictionary,
    Or prints out the whole dictionary created by the count vectoriser object.
    """
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vocabulary_items]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_sum = int(sum_words.sum(axis=1))

    if param == 'dictionary':
        print(words_freq)

    elif param == 'stats':
        x, y, z = [], [], []
        iter = 0
        for k, v in sorted(words_freq, key=lambda x: x[1], reverse=True):
            if iter == k_words:
                break
            else:
                x.append(k)
                y.append((v/total_sum)*100)
                z.append(v)
                iter = iter + 1
        plt.bar(x, y, width=0.6)
        s = str(total_sum)
        plt.title('Frequency Count of the top 10 words. Σ of all Counts: %i' % total_sum)
        plt.ylabel('Frequency in %')
        plt.xlabel('Word')
        plt.show()

        plt.bar(x, z, width=0.6)
        s = str(total_sum)
        plt.title('Word Count of the top 10 words. Σ of all Counts: %i' % total_sum)
        plt.ylabel('Count')
        plt.xlabel('Word')
        plt.show()
    else:
        print('Invalid Parameter.')


def prepare_for_fitness(preprocessed_data):
    """
    :param preprocessed_data: already cleaned and lemmatised data.
    :return: X, count_vect, tdidf_transformer

    Prepare the preprocessed_data into the model. Convert 'text' into 'numbers' (model readable format).
    This function is called only once in the script, thus promising speed improvements while calling the
    preprocessed data by multiple classifiers.

    1.1 Instantiate the vectoriser object with N max_features (its size)
    1.2 Vectorise the preprocessed data. (bag of words model)
    2. Calculate the Term Frequency * Inverse Document Frequency for every word.
    3. <OPTIONAL> Return dictionary statistics.
    """
    print('Vectorising.')

    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    X = count_vect.fit_transform(preprocessed_data)

    # <OPTIONAL>
    # k_words = number of the top K most frequent words.
    # Param: 'dictionary' for printing the dictionary, 'stats' for frequency and count plots.
    return_dictionary_stats(X, count_vect.vocabulary_.items(), k_words=10, param='stats')

    print('Vectorised.')
    print('TD IDF.')
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)
    print('TD IDF done.')
    print('____________________________________________________________________________________')
    print('\n')

    return X, count_vect, tfidf_transformer


def plot_matrix(mat, test_data, classifier_name):
    """
    :param mat: Confusion matrix to be plotted.
    :param test_data: Test_data
    :param classifier_name: Classifier Name

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
    :param classifier_no: 0-1-2 (LINEAR SVM, MLTN NAIVE BAYES, RANDOM FOREST CLASSIFIER)
    :param X1: Sparse Matrix of TD*IDF.
    :param y: Target Names / Categories
    :param count_vect: Vectoriser sent from the prepare_for_fitness method.
    :param tfidf_transformer: Transformer sent from the prepare_for_fitness method
    :return: accuracy score. The accuracy score of a classifier can be computed as
                             (TP + TN) / (TP + FP + FN + TN)
    1. Fit Classifier
    2. Load and Prepare Test Data by vectorising and computing TF*IDF for every token.
    3. Test data in the Fitted Classifier
    4. Report Objective Metrics
    """
    print('Fitting the model.')
    if classifier_no == 0:
        text_clf = svm.SVC(kernel='linear', C=1.0)
        text_clf.fit(X1, y)
    if classifier_no == 1:
        text_clf = MultinomialNB().fit(X1, y)
        text_clf.fit(X1, y)
    if classifier_no == 2:
        text_clf = RandomForestClassifier()
        text_clf = RandomForestClassifier().fit(X1, y)

    print('Results for', text_clf.__class__)
    test_path = '/spam-non-spam-dataset/test-mails'
    test_data = load_files(test_path)
    X_test, y_test = test_data.data, test_data.target


    # Prepare Test Data.
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = text_clf.predict(X_test_tfidf)

    print('The accuracy score: ', np.mean(predicted == y_test))
    print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
    print('confusion matrix', '\n', metrics.confusion_matrix(test_data.target, predicted))
    print('____________________________________________________________________________________')

    # plots the confusion matrix using plot_matrix function
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
if param == 'lingspam':
    print('Classification Started')
    print('Classifiers running on Ling Spam dataset subset')
    path = 'spam-non-spam-dataset/train-mails'
    pd, X, y = preprocessing(path)
    X1, v, t = prepare_for_fitness(pd)
    accuracy = test_classifier(0, X1, y, v, t)
    accuracy = test_classifier(1, X1, y, v, t)
    accuracy = test_classifier(2, X1, y, v, t)
elif param == 'enron':
    print('Classification Started')
    print('Classifiers running on Enron subset')
    path = 'enron/train-data'
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

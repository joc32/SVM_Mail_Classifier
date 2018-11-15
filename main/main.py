import numpy as np
import matplotlib.pyplot as plt
import nltk, re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import preprocessing, metrics, cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import load_files
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def data_cleaner(line):
    '''
    :param line:
    :return:
    '''

    document = re.sub(r'\W', ' ', line)
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    document = re.sub(" \d+", " ", document)  # digits
    # Remove the _ in the document and replace them with ''
    document = document.replace('_', '')
    # Converting to Lowercase
    document = document.lower()
    return document

def lemmatise(line,lmtzr):
    '''
    :param line:
    :param lmtzr:
    :return:
    '''
    list = []
    words = nltk.word_tokenize(line) #splits the line into individual words / tokens
    for item in words:
        item = lmtzr.lemmatize(item) #Lemmatizes every word
        list.append(item)
    return ' '.join(list)

def preprocessing(path):
    '''
    :param path:
    :return:
    '''

    spam_data = load_files(path)
    preprocessed_documents = []

    lmtzr = WordNetLemmatizer() # Instantiates new lemmatiser object.
    X, y = spam_data.data, spam_data.target

    for each in range(0, len(X)):
        document = data_cleaner(str(X[each])) #Data Cleaning
        document = lemmatise(document,lmtzr) #Lemmatise
        preprocessed_documents.append(document)

    return preprocessed_documents,X,y

def test_classifier(classifier_no,documents,X,y):
    '''
    :param classifier_no:
    :param documents:
    :param X:
    :param y:
    :return:
    '''

    print('\n')
    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    X = count_vect.fit_transform(documents)


    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)

    if classifier_no == 0:
        text_clf = MultinomialNB().fit(X,y)
    if classifier_no == 1:
        text_clf = SGDClassifier()
        text_clf = SGDClassifier().fit(X,y)
    if classifier_no == 2:
        text_clf = RandomForestClassifier()
        text_clf = RandomForestClassifier().fit(X,y)

    print('Results for', text_clf.__class__)
    test_path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/test-mails'
    test_data = load_files(test_path)
    X_test, y_test = test_data.data, test_data.target

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = text_clf.predict(X_test_tfidf)

    print('The accuracy score: ',np.mean(predicted == y_test))
    print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
    print('confusion matrix','\n',metrics.confusion_matrix(test_data.target, predicted))

    return np.mean(predicted == y_test)

print('STARTING CLASSIFICATION')

path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/train-mails'
pd,X,y = preprocessing(path)
a = test_classifier(0,pd,X,y)
b = test_classifier(1,pd,X,y)
c = test_classifier(2,pd,X,y)
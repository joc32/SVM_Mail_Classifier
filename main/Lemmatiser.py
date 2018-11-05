import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

lmtzr = WordNetLemmatizer() #Instantiates new lemmatiser object.

file_read = open('cleaned.txt','r')
file_write = open('preprocessed.txt','w')


#Normalises each word.
for line in file_read:
    words = nltk.word_tokenize(line) #splits the line into individual words / tokens
    for item in words:
        item = lmtzr.lemmatize(item) #Lemmatizes every word
        file_write.write(item)
        file_write.write(' ')
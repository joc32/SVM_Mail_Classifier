import os
import re
import string

path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/train-mails/'

lines = []
file_read = open('out.txt','r')
file_write = open('cleaned.txt','w')

for line in file_read:  # for every line in the array of not duplicate lines.
    result = re.sub(r'[^\w\s]','',line)  # regex for punctuation.
    result = result.lower()  # turns every line into lowercase text.
    result = re.sub(" \d+", " ", result)
    result = result.replace("  ","")
    file_write.write(result)

file_read.close()
file_write.close()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import os

# X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
# y = [0,1,0,1,0,1]
#
# clf = svm.SVC(kernel='linear', C = 1.0)
# clf.fit(X,y)
# w = clf.coef_[0]
# print(w)
#
# a = -w[0] / w[1]
#
# xx = np.linspace(0,12)
# yy = a * xx - clf.intercept_[0] / w[1]
#
# h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
#
# plt.scatter(X[:, 0], X[:, 1], c = y)
# plt.legend()
# plt.show()


path = '/Users/joe/PycharmProjects/SVM_Mail_Classifier/main/spam-non-spam-dataset/train-mails/'

file_out = 'out.txt'
file = open(file_out,'w')

# Open the files from the train mails directory and write their contents into one text file.
for filename in os.listdir(path):
    data = open(path+filename)
    for line in (data):
        if line != '\n':
            file.write(line)
file.close()


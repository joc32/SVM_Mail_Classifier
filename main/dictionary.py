import sys
import pickle
import time

# Function that takes an input file, blank dictionary
# and returns a dictionary of words with its frequency.

def dictionary_handler(file_name, my_list):
    file_read = file_name
    wordfreq = my_list

    for line in file_read:
        words = line.split()  # Splits the line into words.
        for word in words:
            if word not in wordfreq:  # if word not in dict, append a word to the dict and set the count to 1.
                wordfreq[word] = 1
            else:
                wordfreq[word] += 1  # if word in the dict, increase the count by one
    return wordfreq


# function that prints a dictionary in ascending order.

def print_sorted_dictionary(my_dict):
    for word in sorted(my_dict, key=my_dict.get, reverse=False):
        print(word, my_dict[word])
        # time.sleep(0.05)


# Description
# There are 3 modes for this script:
# 1. create a vocabulary of word frequencies with an input from an txt file
# 2. extend already created dictionary by a text file
# 3. print the vocabulary
#
# The main vocabulary is not readable by humans since it is in stored in binary mode,
# for efficiency purposes.
#
# The words dictionary and vocabulary have interchangeable meanings as data structure, however,
# the dictionary is usually a subset of the vocabulary which actually is the main ds.


# Start of the script, few command line argument checks.

if len(sys.argv) == 2:
    argument = sys.argv[1]
else:
    argument = sys.argv[1]
    input_file_from_cmd = sys.argv[2]

# block that creates a new vocabulary with a file from command line.
if argument == 'create':
    print('Creating vocaulary')
    wordfreq = {}  # list that holds the frequencies of a every word.
    file_read = open(input_file_from_cmd, 'r')
    dictionary_handler(file_read, wordfreq)  # method call to count the word frequencies of a file

    file_write = open('vocabulary.pkl', 'wb')  # store the dictionary into a pickle file, write mode is binary
    pickle.dump(wordfreq, file_write)
    file_write.close()

# block extends the vocabulary by a file.
if argument == 'extend':
    print('Extending the vocaulary')
    wordfreq = {}
    file_read = open(input_file_from_cmd, 'r')
    dictionary_handler(file_read, wordfreq)

    pkl_file = open('vocabulary.pkl', 'rb')  # open our main vocabulary by from a picket file.
    vocabulary = pickle.load(pkl_file)  # and read it into a variable.
    pkl_file.close()

    #   quick algorithm that checks if a key is in the dictionary that extends the main vocabulary.
    #   description: d0 = dictionary that extends the main vocabulary.
    #                v = main vocabulary.
    for word in wordfreq:
        if word in vocabulary:  # if the key from d0 is in v:
            vocabulary[word] += wordfreq[word]  # update value of a v's key by values of d0's key.
        else:
            vocabulary[word] = 1  # if the key is not present, create a new entry and set the value to 1.

    file_write = open('vocabulary.pkl', 'wb')
    pickle.dump(vocabulary, file_write)  # write the newly extended dictionary into a pickle file, binary mode.
    file_write.close()

#   block that prints the main vocabulary.
if argument == 'print':
    print('Printing the main dictionary')
    time.sleep(3)

    pkl_file = open('vocabulary.pkl', 'rb')  # open the main dictionary from pickle file
    vocabulary = pickle.load(pkl_file)

    print_sorted_dictionary(vocabulary)  # and print it.
import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data_files = {}
    for filename in os.listdir(directory) :
        f = open(os.path.join(directory,filename), "r")
        data_files[filename] = f.read()

    # print(data_files)
    return data_files
    # raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # splitting document words with word_tokenize
    words = nltk.tokenize.word_tokenize(document)

    # list we want after tokenizing
    corr_words = []

    for word in words :
        # converting word to lowercase
        word = word.lower()

        #stopwords like in out aint ... eg. , 'few', 'more', 'most', 'other', 'so
        #string.punctuation gives list '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english") :
            # adding words in list we want which are not stopwords and not string.punctuation
            corr_words.append(word)

    # print(corr_words)
    return corr_words
    # raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # dictionary that maps words to their IDF values
    idf_dict = {}

    # looping over documents
    for key in documents.keys() :
        for word in documents.get(key) :
            # if word already in dictionary we want goto next word
            if word in idf_dict.keys() :
                continue
            num = 0
            for key1 in documents.keys() :
                for word1 in documents.get(key1) :
                    #checking word in the document
                    if word == word1 :
                        #num  = number of document having word
                        num+=1
                        break
            #adding  inverse document frequency value
            #math.log is log with base e or natural log
            idf_dict[word] = math.log(len(documents)/num)

    return idf_dict
    # raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {}
    for key in files.keys() :
        tf_idf[key] = 0
    for key in files.keys() :
        # looping over words in query
        for word in query :
            num = 0
            for word1 in files.get(key) :
                # how many times word appeard in document
                if word1 == word :
                    num+=1
            #tf*idf
            tf_idf[key] += num * (idfs[word])
    # print(tf_idf)
    #sorted files accordng to tf-idf values
    sort_orders = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    # print(sort_orders)
    filenames = []
    for i in sort_orders:
        if n<=0:
            break
        filenames.append(i[0])
        n-=1
    # print(filenames)
    return filenames
    # raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # print(query)
    # print(sentences)
    # print(idfs)
    match_sentences = []
    sentences_idf = {}

    for key in sentences.keys() :
        sentences_idf[key] = [0,0]
    
    for key in sentences.keys() :
        num = len(sentences.get(key))
        #number of times the word occur in sentence
        num1 = 0
        for word in query :
            for word1 in sentences.get(key) :
                #if word in query also in sentence add idf to sum
                if word1 == word :
                    num1+=1
                    sentences_idf[key][0] += idfs[word1]
                    break
        sentences_idf[key][1] = num1/num

    # print(sentences_idf)
    sort_orders = sorted(sentences_idf, key = lambda i: (sentences_idf[i][0], sentences_idf[i][1]), reverse = True)
    # print(sort_orders)
    for i in sort_orders:
        if n<=0:
            break
        match_sentences.append(i)
        n-=1
    # print()
    # print(match_sentences)
    return match_sentences    
    raise NotImplementedError


if __name__ == "__main__":
    main()

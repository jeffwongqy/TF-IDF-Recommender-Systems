import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer

############################### Define Functions ########################
def removePunctuation(docs):
    # define a set of punctuations 
    punctuation = "[!#$%&'()*+,-./:;<=>?@—[\]^_`{|}“~]"
    
    # remove all the punctuations present in a list of documents by using the built-in function "replace ()"
    # store it into a new variable 
    rpunct = docs.replace(punctuation,"", regex = True).replace('"', "", regex = True)
    # return a list of documents with all punctuations removed 
    return rpunct


def removeNumbers(rPunctDocs):
    # define a set of numerical digits
    numbers = "[0123456789]"
    
    # remove all the numerical digits present in a list of documents by using the built-in function "replace ()"
    # store it into a new variable 
    rnumbers = rPunctDocs.replace(numbers, "", regex = True)
    # return a list of documents with all numerical digits removed
    return rnumbers


def tokenization(rnumberDocs):
    
    # use the built-in function - "split ()" to split a phrase, sentences, paragraph or the entire documents into an individual words
    # store it into a new variable 
    tokenizedText = rnumberDocs.str.split(" ", expand = False)
    # return a list of documents with tokenized texts 
    return tokenizedText

    
def removeStopwords(tokenizedDocs):
    # define a set of stopwords
    stopwords = ['i','the','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                     'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                     "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                     'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 
                     'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                     'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
                     'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                     'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                     'further', 'then', 'may', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
                     'any', 'both', 'each', 'few', 'more', 'whose','most', 'other', 'some', 'such', 'no', 'nor', 
                     'not', 'only', 'else', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                     'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
                     've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
                     "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
                     'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                     'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
                     "wouldn't"]
    
    # go through each set of documents with tokenized texts
    for j in range(num_of_docs): 
        # go through each word in the list of stopwords
        for stopword in stopwords:
            # go through each tokenized text in each set of documents  
           for element in tokenizedDocs[j]:
               # if each tokenized text in each set of documents represent the word in the list of stopwords,
               if element == stopword:
                   # remove the respective tokenized text from its respective document by using the built-in function "remove ()"
                   tokenizedDocs[j].remove(element)
    # return a list of documents with stopwords removed.
    return tokenizedDocs


def lemmatizingWords(tokenizedDocs):
    # create a class object for WordNetLemmatizer()
    # and store it into a new variable
    wnl = WordNetLemmatizer()
    # Go through each set of documents with tokenized texts
    for j in range(num_of_docs):
        # go through each tokenized text for each set of documents
        for index in range(len(tokenizedDocs[j])):
            # perform lemmatization process for each tokenized text for each set of documents
            # store it into the same variable
            tokenizedDocs[j][index] = wnl.lemmatize(tokenizedDocs[j][index])
    # return a list of documents with lemmatized texts
    return tokenizedDocs

def compute_TermFreq(wordsDict, bag_of_words):
    # initialize an empty dict
    termFreqDict = dict()
    
    # count the number of terms present in the bags of words
    count_bows = len(bag_of_words)
    
    # iterate a list of words dictionary with value counts using the built-in function "items()"
    for wordOccurrence, val_count in wordsDict.items():
        # compute the term frequency for the respective word occurrence by counting the number of words appear in a document 
        # divided by the total number of terms in the bags of words
        # store the computed term frequency into dict variable
        termFreqDict[wordOccurrence] = val_count/ float(count_bows)
    # return the computed term-frequency
    return termFreqDict


def compute_InverseDocFreq(list_of_docs):
    # create a key-value pairs dict by using the built in function "dict.fromkeys()"
    # initialize all the respective key terms with zero values
    # store it into a new variable
    inverseDocFreqDict = dict.fromkeys(list_of_docs[0].keys(), 0)
    
 
    # go through each list of documents 
    for docs in list_of_docs:
        # go through each key-value pairs dictionary from each list of documents
        for wordOccurrence, val_count in docs.items():
            # if the value count is greater than zero, 
            if val_count > 0:
                # increment the respective key terms by 1 and 
                # store the computed value count with its respective key terms into the dict variable
                inverseDocFreqDict[wordOccurrence] = inverseDocFreqDict[wordOccurrence] + 1
    
    # go through each of the key-value pairs dictionary as computed above
    for wordOccurrence, val_count in inverseDocFreqDict.items():
        # compute the inverse-doc frequency for each respective key terms using IDF-smoothing technique
        inverseDocFreqDict[wordOccurrence] = (np.log((1+len(list_of_docs))/ (1+float(val_count)))) + 1 
    # return the computed idf values
    return inverseDocFreqDict


def compute_TFIDF(tf, idf):
    # initialize an empty dict 
    tf_idfDict = {}
    
    # go through each key-value pairs dict from the list of computed tf values. 
    for wordOccurrence, val in tf.items():
        # compute the tf-idf scores by multiplying the TF-value for the respective word term with the 
        # IDF-value for the same word terms 
        # store the computed tf-idf scores with its respective key-term into the dict 
        tf_idfDict[wordOccurrence] = val * idf[wordOccurrence]
    # return the tf-idf scores with its all the respective key-terms
    return tf_idfDict


################################## main #################################
# display and prompt the user to upload the 5 documents 
num_of_docs = 5
documents = {}
doc_names = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5']

print("Welcome to TFIDF Calculation")
print("***********************************************")
print("Instructions to User:")
print("1) You are required to prepare and upload 5 documents with consecutive chapters.")
for i in range(num_of_docs):
    filepath = input("Enter the filepath of Document {}: ".format(i+1)) # prompt the user for filepath
    text = open(filepath, "r", encoding = "utf-8") # open the content of the file 
    documents[doc_names[i]] = text.read().replace('\n', ' ').lower() # read and store the content from respective file into dict func
    text.close() # close the file after each cycle of reading the text file

# store all the context from respective docs into the Pandas Series
docs = pd.Series(documents)

########################## Preprocessing Steps#########################

# call the function to remove punctuations
# and store it into a new variable
rPunctDocs = removePunctuation(docs)

# call the function to remove numerical digits
# and store it into a new variable
rnumberDocs = removeNumbers(rPunctDocs)

# call the function to perform tokenization process and
# store the returned tokenized texts into a new variable
tokenizedDocs = tokenization(rnumberDocs)

# call the function to remove stopwords
# store the returned tokenized texts with stopwords removed into a new same variable
tokenizedDocs = removeStopwords(tokenizedDocs)

# go through each list of documents with tokenized texts
for j in range(num_of_docs):
    # using the built-in function "filter ()" to remove all the empty list from each respective documents
    # store it into a same variable 
    tokenizedDocs[j] = list(filter(None, tokenizedDocs[j]))

# call the function to perform lemmatization process 
# store the returned lemmatized texts into a new variable
lemmatizedDocs = lemmatizingWords(tokenizedDocs)

# create  a copy of the pre-processed text documents 
# store it into a new variable
bag_of_words = lemmatizedDocs.copy()


########################## Counting Words Occurrence ###################
# extract the unique and non-duplicate feature names
feature_names = list(set().union(bag_of_words[0], bag_of_words[1], bag_of_words[2], bag_of_words[3], bag_of_words[4]))

# initialize an empty dict with zero values for respective terms or words for respective docs
num_of_words = [dict.fromkeys(feature_names, 0) for i in range(num_of_docs)]

# count the number of unique words occurs in the respective documents 
for j in range(num_of_docs):
    numOfWords = num_of_words[j]
    for word in bag_of_words.iloc[j]:
        numOfWords[word] = numOfWords[word] + 1
    num_of_words[j] = numOfWords

# store the respective words count from respective documents into a DataFrame Object
num_of_words_df = pd.DataFrame(num_of_words, index = doc_names)
num_of_words_df_T = pd.DataFrame(num_of_words_df, index = doc_names).T # transpose the dataframe
wordDictIDF = num_of_words_df.copy() # create another copy for IDF computation 

# display the respective words count for the respective documents 
print("\nNumber of Words Occurrence for the Respective Documents: ")
print("**********************************************************")
print(num_of_words_df_T)


############################# TF Computation ############################

# in the list comprehrension, call the function to compute the term-frequency by going through
# each document with a set of bags of words and a list of word dictionary with value counts 
# and store it into a new variable 
termFreq_res = [compute_TermFreq(num_of_words_df.iloc[i], bag_of_words.iloc[i]) for i in range(num_of_docs)]

# store the resultant term-frequency into a new dataframe 
termFreq_df = pd.DataFrame(termFreq_res, index = doc_names)
# transpose the dataframe for the resultant term-frequency
termFreq_df_T = pd.DataFrame(termFreq_res, index = doc_names).T

# display the transpose term frequency results 
print("\nTerm Frequency (TF) for the Respective Keywords Across Documents: ")
print("*********************************************************************")
print(termFreq_df_T)


############################ IDF Computation ###########################

# create a list of documents
list_of_docs = [wordDictIDF.iloc[i] for i in range(num_of_docs)]

# call the function to compute IDF
idf_res = compute_InverseDocFreq(list_of_docs)

# convert the resultant idf from a dictionary into a Pandas Series
idf_series = pd.Series(idf_res)

# display the idf results
print("\nInverse Documents Frequency (IDF) for the Respective Keywords: ")
print("*******************************************************************")
print(idf_series)


############################# TF*IDF Computation #####################

# initialize an empty list
tfidf_res = list()
# go through each list of documents 
for j in range(num_of_docs):
    # call the function to compute TFIDF for each documents
    # append the return TFIDF key- value pairs for each document 
    # and store it into an empty list
    tfidf_res.append(compute_TFIDF(termFreq_df.iloc[j], idf_res))

# convert the tfidf scores into a DataFrame
tfidf_df = pd.DataFrame(tfidf_res, index = doc_names)
# transpose the dataframe
tfidf_df_T = pd.DataFrame(tfidf_res, index = doc_names).T 

# display the transpose tfidf results with its TF-IDF scores for each respective keys across the documents
print("\nTerm Frequency - Inverse Documents Frequency (TF*IDF) Scores for the Respective Keywords Across Documents: ")
print("**************************************************************************************************************")
print(tfidf_df_T)


######################### Ranking & Identification #####################
# sum up the tf-idf scores for each keywords across the documents and sort it in descending order
# store the summed tf-idf scores into a DataFrame
ranking = pd.DataFrame(tfidf_df.sum().sort_values(ascending = False), columns = ['Overall TFIDF Score'])    

# display the summed tfidf scores for each keywords
print("\nSummed TFIDF Scores for Respective Keywords")
print("**********************************************")
print(ranking)


# identification of most relevant/ important keywords in the corpus
# extract the index name for the top 100 important keywords 
top_100 = ranking.head(100).index 
# set the top 100 important keywords into a list
impt_keywords = top_100.tolist() 

# display the top 100 important keywords 
print("\nTop 100 relevant & important keywords in the corpus")
print("**************************************************************************")
print(impt_keywords)


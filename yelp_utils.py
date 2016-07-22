#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import string
import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix

##VARIABLES
SEED_VAL = 200
data_subset = "_0_0_1Percent"
data_frac = 0.001
WORK_DIR = os.getcwd()
YELP_DATA_RAW_DIR = os.path.join(WORK_DIR, "data", "raw")
YELP_DATA_CSV_DIR = os.path.join(WORK_DIR, "data", "csv")
YELP_DATA_WORD_2_VEC_MODEL_DIR = os.path.join(WORK_DIR, "data", "word2vec_model")
YELP_DATA_SPARSE_MATRIX_DIR = os.path.join(WORK_DIR, "data", "sparse_matrix")

pd.options.display.max_columns = 200


### FILE READ WRITE
def make_sure_path_exists(path): 
    '''
    Function to  make a new directory structure if it doesnt exists.
    Created by following: 
    # http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    Input: Directory path
    Output: Creates directories if they dont exist
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        
def getDfInfo(df):
    '''
    Function to display information about pandas data frame
    Input: Pandas data frame
    Output: 
        1. Prints the dimension of dataframe
        2. Prints null percentage in each column of dataframe    
    '''
    nrow = df.shape[0]
# print np.count_nonzero(df.isnull()) / nrow #http://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe
    print "\n*****SHAPE********"
    print df.shape
    print "*****NULL PERCENTAGE*********"
    print df.isnull().sum() / nrow
    df.head()
    

def save_sparse_csr(filename,array):
    '''
    Function to save scipy sparse matrix in numpy uncompressed .npz format
    Created by following: # http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
    Input: 
        filename: The name and path to save the file
        array: scipy.sparse.csr_matrix
    Output: .npz file
    '''
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    '''
    Function to load a file in .npz format into scipy sparse matrix
    Created by following: # http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
    Input: 
        filename: The name and path of the .npz file
    Output: scipy.sparse.csr_matrix 
    '''
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

### NLP
# The maps have been generated following -
# http://stackoverflow.com/questions/11692199/string-translate-with-unicode-data-in-python
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
remove_number_map = dict((ord(char), None) for char in string.digits)

def remove_numbers_in_string(s):
    '''
    Function to remove numbers in a string.
    Input: string
    Output: string
    '''
#     print(type(s))
#     return s.translate(None, string.digits)
    return s.translate(remove_number_map)

def lowercase_remove_punctuation(s):
    '''
    Function to lowercase string and remove punctuation marks
    Input: string
    Output: string
    '''
    s = s.lower()
#     s = s.translate(None, string.punctuation)
    return s.translate(remove_punctuation_map)

NLTK_STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(s):
    '''
    Function to remove stopwords. Stopwords list is used from NLTK package.
    Source: 
    https://github.com/kevin11h/YelpDatasetChallengeDataScienceAndMachineLearningUCSD/blob/master/Yelp%20Predictive%20Analytics.ipynb
    Input: string 
    Output: string
    '''    
    token_list = nltk.word_tokenize(s)
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return ' '.join(filter(exclude_stopwords, token_list))

def filter_out_more_stopwords(token_list, MORE_STOPWORDS):
    '''
    Function to filter out more stopwords
    Source:
    https://github.com/kevin11h/YelpDatasetChallengeDataScienceAndMachineLearningUCSD/blob/master/Yelp%20Predictive%20Analytics.ipynb
    Input: 
        token_list: list of words
        MORE_STOPWORDS: list of stopwords
    Output: 
        list without stopwords
    '''    
    return filter(lambda tok : tok not in MORE_STOPWORDS, token_list)

def stem_token_list(token_list):
    '''
    Function to stem words
    Input: list 
    Output: list 
    '''    
    STEMMER = PorterStemmer()
#     return [STEMMER.stem(tok.decode('utf-8')) for tok in token_list]
    return [STEMMER.stem(tok) for tok in token_list]

def restring_tokens(token_list):
    '''
    Function to convert the the tokenized words to string
    Input: list
    Output: string
    '''    
    return ' '.join(token_list)

def lowercase_and_remove_punctuation_and_remove_numbers_and_tokenize_stem_and_restring(s):
    '''
    Function to lowercase, remove punctuation, remove numbers, stem each token in a string
    Input: string
    Output: string
    '''    
    s = remove_numbers_in_string(s)
    s = lowercase_remove_punctuation(s)
    s = remove_stopwords(s)
    token_list = nltk.word_tokenize(s)
    #token_list = filter_out_more_stopwords(token_list)
    token_list = stem_token_list(token_list)
    return restring_tokens(token_list)


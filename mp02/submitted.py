'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    #raise RuntimeError("You need to write this part!")
    frequency = {}
    for review in train:
        temp = Counter()
        for i in range(0,len(train[review])):
            temp.update(train[review][i])
        frequency[review] = temp

    return frequency

import copy
def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    #raise RuntimeError("You need to write this part!")
    nonstop = copy.deepcopy(frequency)
    for word1 in stopwords:
        for temp in nonstop:
            for word2 in list(nonstop[temp]):
                if word1 == word2:
                    del nonstop[temp][word2]
        
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    #raise RuntimeError("You need to write this part!")
    likelihood = {}
    for review in nonstop:
        temp = {}
        #total_temp = nonstop[review].total()
        total_temp = sum(nonstop[review].values())
        for x in nonstop[review]:
            temp[x] = (nonstop[review][x] + smoothness) / (total_temp + (smoothness * (len(nonstop[review]) + 1)))
        temp['OOV'] = smoothness / (total_temp + (smoothness * (len(nonstop[review]) + 1)))
        likelihood[review] = temp
    
    return likelihood

import math
def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    #raise RuntimeError("You need to write this part!")
    hypotheses = []
    for i in range(0, len(texts)):
        pos = math.log(prior)
        neg = math.log(1 - prior)
        for k in texts[i]:
            if k not in stopwords:
                if k in likelihood['pos']:
                    pos = pos + math.log(likelihood['pos'][k])
                else:
                    pos = pos + math.log(likelihood['pos']['OOV'])
                if k in likelihood['neg']:
                    neg = neg + math.log(likelihood['neg'][k])
                else:
                    neg = neg + math.log(likelihood['neg']['OOV'])
        if pos > neg:
            hypotheses.append('pos')
        elif pos < neg:
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')
    
    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    #raise RuntimeError("You need to write this part!")
    temp_accuracies = []
    for m in range(0, len(priors)):
        temp_row = []
        for n in range(0, len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[n])
            hypotheses = naive_bayes(texts, likelihood, priors[m])
            count_correct = 0
            for(y, yhat) in zip(labels, hypotheses):
                if y == yhat:
                    count_correct += 1
            temp_row.append(count_correct / len(labels))
        temp_accuracies.append(temp_row)


    accuracies = np.array(temp_accuracies)
    return accuracies
                          

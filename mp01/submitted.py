'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    # raise RuntimeError('You need to write this part!')

    maxX1 = 0
    maxX2 = 0
    PjointTemp = []

    for i in range(0, len(texts)): #i is the text
      temp1 = texts[i].count(word0)
      temp2 = texts[i].count(word1)
      if temp1 > maxX1:
        maxX1 = temp1
      if temp2 > maxX2:
        maxX2 = temp2
    
    counter = 0
    for i in range(0, maxX1+1):
      tempList = []
      for j in range(0, maxX2+1):
        for k in range(0, len(texts)):
          if texts[k].count(word0) == i and texts[k].count(word1) == j:
            counter += 1
        tempList.append(counter/len(texts))
        counter = 0
      PjointTemp.append(tempList)
    
    Pjoint = np.array(PjointTemp)
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    #raise RuntimeError('You need to write this part!')
    if index == 0:
      Pmarginal = np.sum(Pjoint, axis = 1)
    else:
      Pmarginal = np.sum(Pjoint, axis = 0)
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    #raise RuntimeError('You need to write this part!')
    PcondTemp = []
    for i in range(0,len(Pjoint)):
      tempList = []
      for j in range(0, len(Pjoint[0])):
        tempList.append(Pjoint[i][j] / Pmarginal[i])
      PcondTemp.append(tempList)
    Pcond = np.array(PcondTemp)
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    #raise RuntimeError('You need to write this part!')
    mu = float(0)
    for i in range(0, len(P)):
      mu += (float(i) * P[i])
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    #raise RuntimeError('You need to write this part!')
    mean = float(mean_from_distribution(P))
    var = float(0)
    for i in range(0, len(P)):
      var += ((float(i) - mean)*(float(i) - mean)*P[i])
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    #raise RuntimeError('You need to write this part!')
    mean0 = float(mean_from_distribution(marginal_distribution_of_word_counts(P,0)))
    mean1 = float(mean_from_distribution(marginal_distribution_of_word_counts(P,1)))
    covar = float(0)
    for i in range(0, len(P)):
      for j in range(0, len(P[1])):
        covar += ((float(i) - mean0)*(float(j) - mean1)*P[i][j])
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    #raise RuntimeError('You need to write this part!')
    expected = float(0)
    for i in range(0, len(P)):
      for j in range(0, len(P[1])):
        expected += f(i, j)*P[i][j]
    return expected
    

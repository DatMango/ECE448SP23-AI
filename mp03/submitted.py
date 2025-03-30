'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''


    #raise RuntimeError('You need to write this part!')
    #distance = np.linalg.norm(train_images-image)
    distance = np.sqrt(np.square(train_images - image).sum(axis = 1))
    indices = distance.argsort()[:k]
    #neighbors_temp = []
    #labels_temp = []
    #for i in range(0,k):
    #    neighbors_temp.append(train_images[indices[i]])
    #    labels_temp.append(train_labels[indices[i]])
    #neighbors = np.array(neighbors_temp)
    #labels = np.array(labels_temp)
    #return neighbors, labels
    return train_images[indices], train_labels[indices]


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    
    #raise RuntimeError('You need to write this part!')
    hypotheses_temp = []
    scores_temp = []
    for m in range(0,len(dev_images)):
        temp = k_nearest_neighbors(dev_images[m], train_images, train_labels, k)
        if temp[1].sum() > (k - temp[1].sum()):
            hypotheses_temp.append(1)
            scores_temp.append(temp[1].sum())
        else:
            hypotheses_temp.append(0)
            scores_temp.append(k - temp[1].sum())
        
    return np.array(hypotheses_temp), np.array(scores_temp)


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    #raise RuntimeError('You need to write this part!')
    confusions = np.array([[0, 0],[0, 0]])
    for m in range(0, len(hypotheses)):
        if hypotheses[m] == references[m]:
            if hypotheses[m] == 1:
                confusions[1][1] += 1
            else:
                confusions[0][0] += 1
        else:
            if hypotheses[m] == 1:
                confusions[0][1] += 1
            else:
                confusions[1][0] += 1
    accuracy = (confusions[1][1] + confusions[0][0]) / (len(hypotheses))
    f1 = 2 / (((confusions[1][1] + confusions[1][0]) / (confusions[1][1])) + ((confusions[1][1] + confusions[0][1]) / (confusions[1][1])))
    return confusions, accuracy, f1

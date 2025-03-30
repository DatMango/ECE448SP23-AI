'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #raise NotImplementedError("You need to write this part!")
    predictions = [] # rv
    tag_w = dict()
    all_words = dict()
    
    for sentence in train:
        for w, t in sentence: # w = word, t = tag
            if w not in all_words:
                all_words[w] = dict()

            if t not in all_words[w]:
                all_words[w][t] = 1
            else:
                all_words[w][t] += 1

    max_tag = None
    for w in all_words.keys():
        counter = all_words[w]
        max_count = 0
        for t in counter.keys():
            if counter[t] > max_count:
                max_count = counter[t]
                max_tag = t
            if t not in tag_w:
                tag_w[t] = 0
            tag_w[t] += counter[t]
        all_words[w] = max_tag
    
    max_count = 0
    for t in tag_w:
        if tag_w[t] > max_count:
            max_count = tag_w[t]
            max_tag = t

    for sentence in test:
        pred_sentence = []
        for word in sentence:
            if word not in all_words:
                pred_sentence.append((word, max_tag))
            else:
                pred_sentence.append((word, all_words[word]))
        predictions.append(pred_sentence)
    return predictions
      


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #raise NotImplementedError("You need to write this part!")
    smoothness = 0.00001
    init_prob = 1
    predictions = []

    # Count occurrences of tag pairs
    tag_pairs = dict() # a
    for sentence in train:
        for i in range(0, len(sentence) - 1):
            prev_tag = sentence[i][1] # tag occurs first
            fol_tag = sentence[i+1][1] # tag occurs second
            if prev_tag not in tag_pairs:
                tag_pairs[prev_tag] = dict()

            if fol_tag not in tag_pairs[prev_tag]:
                tag_pairs[prev_tag][fol_tag] = 1
            else:
                tag_pairs[prev_tag][fol_tag] += 1
    
    # Computer smoothed probabilities (tag pairs)
    for prev_tag in tag_pairs:
        # Ref code from MP2
        # total_temp = sum(nonstop[review].values())
        # for x in nonstop[review]:
        #    temp[x] = (nonstop[review][x] + smoothness) / (total_temp + (smoothness * (len(nonstop[review]) + 1))) # EQ
        # temp['OOV'] = smoothness / (total_temp + (smoothness * (len(nonstop[review]) + 1)))

        # Ref Eq
        # P(x|y) = ((#token of word x in texts of class y) + k) / ((# tokens of any word in texts of class y) + k*(# of word types + 1))
        total_words = sum(tag_pairs[prev_tag].values())
        for fol_tag in tag_pairs[prev_tag]:
            tag_pairs[prev_tag][fol_tag] = (tag_pairs[prev_tag][fol_tag] + smoothness) / (total_words + (smoothness * (len(tag_pairs[prev_tag]) + 1)))
        tag_pairs[prev_tag]["OOV"] = smoothness / (total_words + (smoothness * (len(tag_pairs[prev_tag]) + 1)))

    # Count occurrences of tag/word pairs & Smoothed Probabilities 
    tag_word_pairs = dict() # b
    for sentence in train:
        for i in range(0, len(sentence) - 1):
            for w, t in sentence:
                if t not in tag_word_pairs:
                    tag_word_pairs[t] = dict()

                if w not in tag_word_pairs[t]:
                    tag_word_pairs[t][w] = 1
                else:
                    tag_word_pairs[t][w] += 1
    for t in tag_word_pairs:
        total_words = sum(tag_word_pairs[t].values())
        for w in tag_word_pairs[t]:
            tag_word_pairs[t][w] = (tag_word_pairs[t][w] + smoothness) / (total_words + (smoothness * (len(tag_word_pairs[t]) + 1)))
        tag_word_pairs[t]["OOV"] = smoothness / (total_words + (smoothness * (len(tag_word_pairs[t]) + 1)))

    # Trellis (Viterbi)
    for sentence in test: # sentence = n
        # Initialize v
        v = [dict() for i in range(len(sentence))]
        v_tag = [dict() for i in range(len(sentence))]
        # v = [dict()] * len(sentence)
        # v_tag = [dict()] * len(sentence)
        # for n in range(0, len(sentence) - 1):
        #     v[n] = dict()
        #     v_tag[n] = dict()
        
        # Initialization
        for i in tag_word_pairs: # b
            if i == 'START':
                v[0][i] = 1 #log(init_prob)
            else:
                v[0][i] = 0 #log((1.0 - init_prob) / len(tag_word_pairs.keys()))
        
        # Iteration
        for j in range(1, len(sentence)):
            w = sentence[j]
            for t in tag_word_pairs: # b
                max_tag = None
                max_prob = -float("inf")
                for prev_tag in tag_pairs:
                    if t not in tag_pairs[prev_tag]:
                        if w not in tag_word_pairs[t]:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag]["OOV"] * tag_word_pairs[t]["OOV"]
                        else:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag]["OOV"] * tag_word_pairs[t][w]
                    else:
                        if w not in tag_word_pairs[t]:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag][t] * tag_word_pairs[t]["OOV"]
                        else:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag][t] * tag_word_pairs[t][w]
                    if temp > max_prob:
                        max_prob = temp
                        max_tag = prev_tag
                v[j][t] = max_prob
                v_tag[j][t] = max_tag

        # Termination
        last_tag = "END"
        tags = ["END"]
        for i in range(len(sentence) - 2, -1, -1):
            last_tag = v_tag[i + 1][last_tag]
            tags.append(last_tag)
        tags.reverse()
        predictions.append(list(zip(sentence, tags)))
    return predictions


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #raise NotImplementedError("You need to write this part!")
    smoothness = 0.00001
    init_prob = 1
    predictions = []
    hapax_smooth = 0.001

    # Count occurrences of tag pairs
    tag_pairs = dict() # a
    for sentence in train:
        for i in range(0, len(sentence) - 1):
            prev_tag = sentence[i][1] # tag occurs first
            fol_tag = sentence[i+1][1] # tag occurs second
            if prev_tag not in tag_pairs:
                tag_pairs[prev_tag] = dict()

            if fol_tag not in tag_pairs[prev_tag]:
                tag_pairs[prev_tag][fol_tag] = 1
            else:
                tag_pairs[prev_tag][fol_tag] += 1
    
    # Computer smoothed probabilities (tag pairs)
    for prev_tag in tag_pairs:
        # Ref code from MP2
        # total_temp = sum(nonstop[review].values())
        # for x in nonstop[review]:
        #    temp[x] = (nonstop[review][x] + smoothness) / (total_temp + (smoothness * (len(nonstop[review]) + 1))) # EQ
        # temp['OOV'] = smoothness / (total_temp + (smoothness * (len(nonstop[review]) + 1)))

        # Ref Eq
        # P(x|y) = ((#token of word x in texts of class y) + k) / ((# tokens of any word in texts of class y) + k*(# of word types + 1))
        total_words = sum(tag_pairs[prev_tag].values())
        for fol_tag in tag_pairs[prev_tag]:
            tag_pairs[prev_tag][fol_tag] = (tag_pairs[prev_tag][fol_tag] + smoothness) / (total_words + (smoothness * (len(tag_pairs[prev_tag]) + 1)))
        tag_pairs[prev_tag]["OOV"] = smoothness / (total_words + (smoothness * (len(tag_pairs[prev_tag]) + 1)))

    # Count occurrences of tag/word pairs & Smoothed Probabilities 
    tag_word_pairs = dict() # b
    for sentence in train:
        for i in range(0, len(sentence) - 1):
            for w, t in sentence:
                if t not in tag_word_pairs:
                    tag_word_pairs[t] = dict()

                if w not in tag_word_pairs[t]:
                    tag_word_pairs[t][w] = 1
                else:
                    tag_word_pairs[t][w] += 1
    # Calculate P(T|hapax)
    hapax = dict()
    hapax_count = hapax_smooth
    for t in tag_word_pairs:
        for w in tag_word_pairs[t]:
            if tag_word_pairs[t][w] == 1:
                hapax_count += (1 + hapax_smooth)
    for t in tag_word_pairs:
        temp_hapax_count = hapax_smooth
        for w in tag_word_pairs[t]:
            if tag_word_pairs[t][w] == 1:
                temp_hapax_count += (1 + hapax_smooth)
        hapax[t] = temp_hapax_count / hapax_count
    # Smooth probabilities
    for t in tag_word_pairs:
        new_smoothness = smoothness * hapax[t]
        total_words = sum(tag_word_pairs[t].values())
        for w in tag_word_pairs[t]:
            tag_word_pairs[t][w] = (tag_word_pairs[t][w] + new_smoothness) / (total_words + (new_smoothness * (len(tag_word_pairs[t]) + 1)))
        tag_word_pairs[t]["OOV"] = new_smoothness / (total_words + (new_smoothness * (len(tag_word_pairs[t]) + 1)))

    # Trellis (Viterbi)
    for sentence in test: # sentence = n
        # Initialize v
        v = [dict() for i in range(len(sentence))]
        v_tag = [dict() for i in range(len(sentence))]
        # v = [dict()] * len(sentence)
        # v_tag = [dict()] * len(sentence)
        # for n in range(0, len(sentence) - 1):
        #     v[n] = dict()
        #     v_tag[n] = dict()
        
        # Initialization
        for i in tag_word_pairs: # b
            if i == 'START':
                v[0][i] = 1 #log(init_prob)
            else:
                v[0][i] = 0 #log((1.0 - init_prob) / len(tag_word_pairs.keys()))
        
        # Iteration
        for j in range(1, len(sentence)):
            w = sentence[j]
            for t in tag_word_pairs: # b
                max_tag = None
                max_prob = -float("inf")
                for prev_tag in tag_pairs:
                    if t not in tag_pairs[prev_tag]:
                        if w not in tag_word_pairs[t]:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag]["OOV"] * tag_word_pairs[t]["OOV"]
                        else:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag]["OOV"] * tag_word_pairs[t][w]
                    else:
                        if w not in tag_word_pairs[t]:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag][t] * tag_word_pairs[t]["OOV"]
                        else:
                            temp = v[j - 1][prev_tag] * tag_pairs[prev_tag][t] * tag_word_pairs[t][w]
                    if temp > max_prob:
                        max_prob = temp
                        max_tag = prev_tag
                v[j][t] = max_prob
                v_tag[j][t] = max_tag

        # Termination
        last_tag = "END"
        tags = ["END"]
        for i in range(len(sentence) - 2, -1, -1):
            last_tag = v_tag[i + 1][last_tag]
            tags.append(last_tag)
        tags.reverse()
        predictions.append(list(zip(sentence, tags)))
    return predictions



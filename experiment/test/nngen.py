# encoding=utf-8

import os
import time
import fire
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu

def load_data(path):
    """load lines from a file"""
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines


def find_mixed_nn(simi, diffs, test_diff, bleu_thre :int =5) -> int:
    """Find the nearest neighbor using cosine simialrity and bleu score"""
    candidates = simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = 0
    for j in candidates:
        score = sentence_bleu([diffs[j].split()], test_diff.split())
        if score > max_score:
            max_score = score
            max_idx = j
    return max_idx


def find_nn(simi) -> int:
    """Find the nearest neighbor"""
    max_idx = simi.argsort()[-1]
    return max_idx

def nngen(train_diffs :List[str], train_msgs :List[str], test_diffs :List[str],
    type :"'mixed': cosine + bleu, 'cos': cosine only" ='mixed',
    bleu_thre :"how many candidates to consider before calculating bleu_score" =5) -> List[str]:
    """NNGen
    NOTE: currently, we haven't optmize for large dataset. You may need to split the 
    large training set into several chunks and then calculate the similarities between
    train set and test set to speed up the algorithm. You may also leverage GPU through
    pytorch or other libraries.
    """
    if type not in ["mixed", "cos"]:
        raise ValueError('Wrong tyoe for nngen.')
    counter = CountVectorizer()
    train_matrix = counter.fit_transform(train_diffs)
    # print(len(counter.vocabulary_))
    test_matrix = counter.transform(test_diffs)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_msgs = []
    for idx, test_simi in enumerate(similarities):
        if (idx + 1) % 100 == 0:
            print(idx+1)
        if type == 'mixed':
            max_idx = find_mixed_nn(test_simi, train_diffs, test_diffs[idx], bleu_thre)
        else:
            max_idx = find_nn(test_simi)
        test_msgs.append(train_msgs[max_idx])
    return test_msgs


def main(train_diff_file :str, train_msg_file :str, test_diff_file :str):
    """Run NNGen with default given dataset using default setting"""
    start_time = time.time()
    test_dirname = os.path.dirname(test_diff_file)
    test_basename = os.path.basename(test_diff_file)
    out_file =  "./nngen." + test_basename.replace('.diff', '.msg')
    train_diffs = load_data(train_diff_file)
    train_msgs = load_data(train_msg_file)
    test_diffs = load_data(test_diff_file)
    out_msgs = nngen(train_diffs, train_msgs, test_diffs)
    with open(out_file, 'w') as out_f:
        out_f.write("\n".join(out_msgs) + "\n")
    time_cost = time.time() -start_time
    print("Done, cost {}s".format(time_cost))


if __name__ == "__main__":
    fire.Fire({
        'main':main   
    })

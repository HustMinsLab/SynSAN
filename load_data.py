# -*- coding:utf-8 -*-
import os
import numpy as np
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import word_tokenize

os.environ['STANFORD_PARSER'] = 'E:/software/StanfordNLP/jars/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'E:/software/StanfordNLP/jars/stanford-parser-3.9.1-models.jar'
os.environ['JAVAHOME'] = "D:/software/Java/bin/java.exe"

parser = StanfordDependencyParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')


def load_data():
    """
    load dataset, label
    """
    # load text data
    docs = []
    with open('dataset/text_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            docs.append([word.lower() for word in word_tokenize(line.strip('\n'))])

    # obtain syntactic relation pairs
    parse_result = [[list(parse.triples()) for parse in graphs][0] for graphs in parser.parse_sents(docs)]
    pair_data = [[(tri[0][0], tri[2][0]) for tri in result] for result in parse_result]

    # load vote data
    label = []
    with open('dataset/vote.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vote_value = [float(vote) for vote in line.strip('\n').split(' ')]
            label.append(np.array(vote_value)/sum(vote_value))
    label = np.array(label).astype(np.float32)

    return pair_data, label, docs, parse_result


# if __name__ == '__main__':
#     pair_data, label, parse_result = load_data()

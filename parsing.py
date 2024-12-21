import argparse
from numpy.matrixlib.defmatrix import matrix
from tqdm import tqdm
import numpy as np
import pickle
from transformers import (BertModel, BertTokenizer,
                          RobertaModel,RobertaTokenizer,
                          BertForSequenceClassification, BertConfig,
                          RobertaForSequenceClassification, RobertaConfig)
from dependency import dep_parsing
import os
import pickle
    
def tree_to_matrix(trees, self_loop=False):
    matrices = []
    for tree in trees:
        n = 0
        for u, v in tree:
            n = max(n, max(u, v) + 1)
        matrix =[ [0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1
        if self_loop:
            for u, v in tree:
                matrix[u][v] = 1
        matrices.append(matrix)
    return matrices
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--matrix_dir',
                        type=str,
                        required=True)
    parser.add_argument('--decoder',
                        type=str,
                        choices=['eisner', 'cle'],
                        default='cle')
    parser.add_argument('--subword',
                        type=str,
                        choices=['first', 'avg', 'max'],
                        default='avg')
    parser.add_argument('--root',
                        type=bool,
                        default=True)
    
    parser.add_argument('--output_dir',
                        type=str,
                        default='results')
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for matrix in os.listdir(args.matrix_dir):
        if matrix.endswith('.pkl') and not matrix.endswith('_tree.pkl'):
            args.matrix = os.path.join(args.matrix_dir, matrix)
            trees, results = dep_parsing.decoding(args)
            with open(os.path.join(args.output_dir, matrix.replace('.pkl', '_tree.pkl')), 'wb') as f:
                pickle.dump(trees, f)
    
        
    
    
    
    
    
    
    
    
    
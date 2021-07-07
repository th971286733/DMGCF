'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--seed',  type=int, default=2020) 
    parser.add_argument('--utopk',  type=int, default=6 ) 
    parser.add_argument('--itopk',  type=int, default=6 ) 
    parser.add_argument('--wtu',  type=int, default=1 )     
    parser.add_argument('--wti',  type=int, default=1 ) 
    parser.add_argument('--UItype',  type=int, default=3 ) 
    parser.add_argument('--epoch',  type=int, default=60 ) 
    parser.add_argument('--batch_size',  type=int, default=2048 ) 
    parser.add_argument('--lr',  type=float, default=0.001 ) 
  
    parser.add_argument('--dataset', nargs='?', default='100k',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
  
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[32]',
                        help='Output sizes of every layer')
    parser.add_argument('--lysz', type=int, default=2)

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--dk', type=float, default=0.2,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--info', type=str, default="" ) 
    parser.add_argument('--decay', type=float, default=0.001 ) 
    parser.add_argument('--mode', type=str, default="O" )
    return parser.parse_args()

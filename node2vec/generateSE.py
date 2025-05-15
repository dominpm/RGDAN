import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = False 
p = 1
q = 2
num_walks = 1000
walk_length = 100
dimensions = 64
window_size = 10
iter = 50
Adj_file = '../CUSTOM_P12_Q1/Adj(CUSTOM).txt'
SE_file = '../CUSTOM_P12_Q1/SE(CUSTOM).txt'

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size = dimensions,
        epochs=iter)
    model.wv.save_word2vec_format(output_file)
	
    return
    
nx_G = read_graph(Adj_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)

import nltk
import os
import torch

from os.path import dirname, join as pjoin
from models import InferSent
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# helper code to load infersent dataset
# modified based on the code in from https://github.com/facebookresearch/InferSent
def load_infersent_model():
    file_path = dirname(os.path.realpath(__file__))
    MODEL_PATH = f'{file_path}\\encoder\\infersent2.pkl'
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = f'{file_path}\\fastText\\crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)   

    return infersent

# helper code to load infersent dataset
# modified based on the code in from https://github.com/facebookresearch/InferSent
def build_encoding(sentences):
    infersent = load_infersent_model()
    infersent.build_vocab(sentences, tokenize=True)
    embeddings = infersent.encode(sentences, tokenize=True)
    # infersent.visualize('Yes I can call you?.', tokenize=True)
    
    cos_similarity = cosine_similarity(embeddings)
    euclidean_distance_similarity = euclidean_distances(embeddings)
    
    return cos_similarity, euclidean_distance_similarity


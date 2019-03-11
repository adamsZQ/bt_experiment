import argparse
import numpy as np

import torch
from torch import nn

from bilstm_crf.BiLSTM_CRF_nobatch import prepare_sequence, bilstm_train
from data.glove import Glove_Embeddings
from data.training_data import get_training_data

if __name__ == '__main__':
    # add parser to get prefix
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix") # data and model prefix
    parser.add_argument("--model") # choose model(lstm/bilstm)
    args = parser.parse_args()

    HIDDEN_DIM = 4
    FILE_PREFIX = args.prefix
    model_type = args.model

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    # get word embeddings
    glove_embeddings = Glove_Embeddings(FILE_PREFIX)
    glove_embeddings.words_expansion()
    word_embeddings = glove_embeddings.task_embeddings
    word2id = glove_embeddings.task_word2id
    tag2id = glove_embeddings.task_tag2id

    # get training data
    sentences_data, tag_data = (get_training_data(FILE_PREFIX))

    # sentence data -> index
    sentences_prepared = prepare_sequence(sentences_data, word2id)
    tag_prepared = prepare_sequence(tag_data, tag2id)

    # initialize embedding
    word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(np.array(word_embeddings)))

    if model_type == 'lstm':
        pass
    elif model_type == 'bilstm':
        bilstm_train(word2id,
              tag2id,
              word_embeddings,
              word_embeds,
              device,
              FILE_PREFIX,
              sentences_prepared,
              tag_prepared,)
    else:
        print('please input model name correcyly(lstm/bilstm)')
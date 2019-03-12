import argparse
import numpy as np

import torch
from torch import nn

from bilstm_crf.BiLSTM_CRF_nobatch import bilstm_train
from data.glove import Glove_Embeddings
from data.training_data import get_training_data
from lstm.lstm import lstm_train


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING_TAG = "<PAD>"
UNK_TAG = '<UNK>'


# sentences - > padded index sequence
def prepare_sequence(sentences, item2id, boundary_tags=False):
    sentences_idx = []
    for sentence in sentences:
        sentence_idx = []
        if boundary_tags:
            sentence_idx.append(item2id[START_TAG])
            for w in sentence:
                sentence_idx.append(item2id[w])
            sentence_idx.append(item2id[STOP_TAG])

            sentences_idx.append(sentence_idx)
        else:
            sentences_idx.append([item2id[w] for w in sentence])

    return sentences_idx


if __name__ == '__main__':
    # add parser to get prefix
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix") # data and model prefix
    parser.add_argument("--model") # choose model(lstm/bilstm)
    parser.add_argument("--boundary_tags") # add START END tag
    args = parser.parse_args()

    HIDDEN_DIM = 4
    FILE_PREFIX = args.prefix
    model_type = args.model
    boundary_tags = args.boundary_tags

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    # # TODO test
    if FILE_PREFIX is None:
        FILE_PREFIX = '/path/bt'
    if model_type is None:
        model_type = 'lstm'
    if boundary_tags is None:
        boundary_tags = True

    # get word embeddings
    glove_embeddings = Glove_Embeddings(FILE_PREFIX)
    glove_embeddings.words_expansion()
    word_embeddings = glove_embeddings.task_embeddings
    word2id = glove_embeddings.task_word2id
    tag2id = glove_embeddings.task_tag2id

    # get training data
    sentences_data, tag_data = (get_training_data(FILE_PREFIX))

    # sentence data -> index
    sentences_prepared = prepare_sequence(sentences_data, word2id, boundary_tags)
    tag_prepared = prepare_sequence(tag_data, tag2id, boundary_tags)

    # initialize embedding
    word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(np.array(word_embeddings)))

    if model_type == 'lstm':
        lstm_train(word2id,
                     tag2id,
                     word_embeddings,
                     word_embeds,
                     device,
                     FILE_PREFIX,
                     sentences_prepared,
                     tag_prepared, )
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
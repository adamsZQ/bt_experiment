# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.glove import Glove_Embeddings
from data.training_data import get_training_data
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

torch.manual_seed(1)

#####################################################################
# Helper functions to make the code more readable.


# sentences - > padded index sequence
def prepare_sequence(sentences, word2id):
    sentences_idx = []
    for sentence in sentences:
        sentences_idx.append([word2id[w] for w in sentence])

    sent_lengths = [len(sent) for sent in sentences_idx]

    pad_token = word2id[PADDING_TAG]
    longest_sent = max(sent_lengths)
    batch_size = len(sentences_idx)
    padded_sent = np.ones((batch_size, longest_sent)) * pad_token

    for i, sent_len in enumerate(sent_lengths):
        sequence = sentences_idx[i]
        padded_sent[i, 0:sent_len] = sequence[:sent_len]

    return padded_sent


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def val(X_val, y_val):
    val_loss = model.neg_log_likelihood(X_val, y_val)  # 计算loss
    return val_loss.item()
#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        print('alpha',alpha)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = word_embeds(sentence).float()
        lstm_out, self.hidden = self.lstm(embeds)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        print('lstm-feats',lstm_feats)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # TODO add batch
        score = torch.zeros(1)
        # tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        """batch """
        # for enu in range(feats.size()[0]):
        #     tag_enu = tags[enu]
        #     feat_enu = feats[enu]
        for i, feat in enumerate(feats):
            print('feat', feats)
            print('tags', tags)
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feats[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        print(forward_score)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################
# Run training


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING_TAG = "<PAD>"
UNK_TAG = '<UNK>'
HIDDEN_DIM = 4
FILE_PREFIX = '/path/bt'

# get word embeddings
glove_embeddings = Glove_Embeddings(FILE_PREFIX)
glove_embeddings.words_expansion()
word_embeddings = glove_embeddings.task_embeddings
word2id = glove_embeddings.task_word2id
tag2id = glove_embeddings.task_tag2id

# get training data
sentences_data, tag_data = (get_training_data(FILE_PREFIX))

# sentence data -> index
sentences_padded = prepare_sequence(sentences_data, word2id)
tag_padded = prepare_sequence(tag_data, tag2id)

# initialize embedding
word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(np.array(word_embeddings)))
word_embeds.padding_idx = word2id[PADDING_TAG]

# split train val test
X_train, X_test, y_train, y_test = train_test_split(sentences_padded, tag_padded, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
X_train = torch.from_numpy(X_train).long()
X_test = torch.from_numpy(X_test).long()
X_val = torch.from_numpy(X_val).long()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
y_val = torch.from_numpy(y_val).long()

# dataloader
batch_size = 1
torch_dataset = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

model = BiLSTM_CRF(len(word2id), tag2id, word_embeddings[0].size, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
epoch = 10000
for num_epochs in range(epoch):
    for step, (batch_x, batch_y) in enumerate(loader):

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(batch_x, batch_y)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())
        if num_epochs % 20 == 0:
            val_loss = val(X_val, y_val)
            print('Epoch[{}/{}]'.format(num_epochs, epoch) + 'loss: {:.6f}'.format(
                loss.item()) + 'validation loss: {:.6f}'.format(val_loss))
            # print()

            # if val_loss < minimum:
            #     print('successssssssssssssssssssssssssssssss')
            #     joblib.dump(belief_tracker, 'model/model_tracker_v2_' + str(val_loss) + '.m')
            #     return




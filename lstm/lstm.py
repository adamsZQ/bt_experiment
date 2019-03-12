import json
import os

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import sys

import visdom as visdom
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.path.join(os.getcwd() + '/'))


class FacetTracker(nn.Module):
    def __init__(self, tag_to_ix, hidden_size, embedding_dim):
        super(FacetTracker, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = len(tag_to_ix)

        # batch * seq_len * input_size x * 4 * 4287
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        # self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)

        # 通过线性层完成从隐层空间到slot value空间的映射
        self.out = nn.Linear(self.hidden_size, self.output_size, True)

        # softmax函数完成最后的归一, TODO 交叉熵验证不需要softmax层
        self.softmax = nn.Softmax(dim=1)

        # self.hidden = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(device)

    def forward(self, word_embeds, sentence):
        embeds = word_embeds(sentence).float().view(1, len(sentence), -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_size)
        lstm_feats = self.out(lstm_out)
        lstm_sofmax = self.softmax(lstm_feats)
        return lstm_feats, lstm_sofmax


def save_model(model, file_prefix=None, file_name=None, val_loss='None', best_loss='None', enforcement = False):
    # Save model
    try:
        if enforcement or val_loss == 'None' or best_loss == 'None':
            file_path = '{}{}_{}.pkl'.format(file_prefix, file_name, 'enforcement')
            torch.save(model, file_path)
            print('enforcement save:', file_path)

        elif val_loss != 'None' and best_loss != 'None' and ~enforcement:
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            if is_best:
                file_path = '{}{}_{:.4f}.pkl'.format(file_prefix, file_name, best_loss)
                torch.save(model, file_path)
            return best_loss
    except Exception as e:
        # if error, save model in default path
        print(e)
        file_path = '{}{}.pkl'.format(os.getcwd(), '/default')
        print('default save:', file_path)
        torch.save(model, file_path)


def val(model, word_embeds, device, X_val, y_val):
    predict_list = []
    target_list = []
    for sentence, tags in zip(X_val,y_val):
        sentence = torch.tensor(sentence).long().to(device)
        lstm_feats, lstm_sofmax = model(word_embeds, sentence)
        predict = torch.argmax(lstm_sofmax, dim=1)
        predict_list.append(predict.tolist())
        target_list.append(tags)

    binarizer = MultiLabelBinarizer()
    binarizer.fit_transform([[x for x in range(model.output_size)]])
    target_list = binarizer.transform(target_list)
    predict_list = binarizer.transform(predict_list)

    accuracy = accuracy_score(target_list, predict_list)
    f1 = f1_score(target_list, predict_list, average="samples")
    precision = precision_score(target_list, predict_list, average="samples")
    recall = recall_score(target_list, predict_list, average="samples")

    return accuracy, precision, recall, f1


def lstm_train(word2id,
          tag2id,
          word_embeddings,
          word_embeds,
          device,
          model_prefix,
          sentences_prepared,
          tag_prepared,
          HIDDEN_DIM=4,):

    model = FacetTracker(tag2id, HIDDEN_DIM,word_embeddings[0].size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    word_embeds = word_embeds.to(device)

    X_train, X_test, y_train, y_test = train_test_split(sentences_prepared, tag_prepared, test_size=0.1, random_state=0, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    # initialize visdom
    viz = visdom.Visdom()
    win = viz.scatter(X=np.asarray([[0, 0]]))
    epoch = 1000
    best_loss = 1e-1
    model_prefix = model_prefix
    file_name = 'lstm'
    for num_epochs in range(epoch):
        # for step
        for sentence, tags in zip(X_train, y_train):
            # Step 3. Run our forward pass.
            sentence = torch.tensor(sentence).long().to(device)
            # torch.unsqueeze(sentence, 0)
            tags = torch.tensor(tags).long().to(device)

            lstm_feats, lstm_sofmax = model(word_embeds, sentence)
            # cross entropy already has softmax
            loss = criterion(lstm_feats, tags)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss.item())
        if num_epochs % 1 == 0:
            accuracy, precision, recall, f1 = val(model, word_embeds, device, X_val, y_val)

            # draw loss line
            viz.scatter(X=np.array([[num_epochs, loss.tolist()]]),
                        name='train',
                        win=win,
                        update='append')

            viz.scatter(X=np.array([[num_epochs, f1]]),
                        name='validate',
                        win=win,
                        update='append')

            print('Epoch[{}/{}]'.format(num_epochs, epoch) + 'loss: {:.6f}'.format(
                loss.item()) +
                  'accuracy_score: {:.6f}'.format(accuracy) +
                  'precision_score: {:.6f}'.format(precision) +
                  'recall_score: {:.6f}'.format(recall) +
                  'f1_score: {:.6f}'.format(f1))

            best_loss = save_model(model, model_prefix, file_name, 1 - f1, best_loss)

    save_model(model_prefix, file_name, enforcement=True)




import math
import torch
import numpy as np
import torch.nn as nn
# import toch.nn.functional as F
from constants import *

class LSTM(nn.Module):
    def __init__(self,vocab_size,glove,word2index):
        super().__init__()
        self.embedding, num_embeddings, embedding_dim = self.create_emb_layer(vocab_size,glove,word2index)
        self.lstm = nn.LSTM(embedding_dim, HIDDEN_SIZE,batch_first = True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dropout = nn.Dropout(LSTM_DROPOUT)

    def create_emb_layer(self,vocab_size,glove,word2index):
        matrix_len = vocab_size
        weights_matrix = np.zeros((matrix_len, 201))

        for i, word in enumerate(word2index.keys()):
            try:
                weights_matrix[i] = glove[word]
            except KeyError:
                weights_matrix[i] = glove["unk"]

        weights_matrix = torch.from_numpy(weights_matrix)
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        output2 = self.fc(hidden.squeeze(0))
        output3 = self.dropout(output2)

        return output3

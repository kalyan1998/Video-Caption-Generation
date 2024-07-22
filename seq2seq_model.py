import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import pickle

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, hidden_state = self.gru(input)

        return output, hidden_state

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3, beam_width=3):
        super(Decoder, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.beam_width = beam_width

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def beam_search(self, encoder_last_hidden_state, encoder_output, beam_width=3, max_seq_len=28):
            _, batch_size, _ = encoder_last_hidden_state.size()
            decoder_hidden = encoder_last_hidden_state
            decoder_input = Variable(torch.ones(batch_size, 1)).long().cuda()

            beam_scores = torch.zeros((batch_size, beam_width)).cuda()
            beam_seq = torch.ones((batch_size, beam_width, max_seq_len)).long().cuda() * self.pad_token_id

            beam_seq[:, :, 0] = self.start_token_id

            encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_width, 1, 1)
            encoder_output = encoder_output.view(batch_size * beam_width, -1, encoder_output.size(-1))

            for t in range(max_seq_len - 1):
                if t == 0:
                    context = self.attention(decoder_hidden, encoder_output[:, 0, :])
                else:
                    context = self.attention(decoder_hidden.view(batch_size, beam_width, -1)[:, 0, :], encoder_output[:, 0, :])

                gru_input = torch.cat((self.embedding(decoder_input).squeeze(1), context), dim=1).unsqueeze(1)
                output, decoder_hidden = self.gru(gru_input, decoder_hidden)
                output_logits = self.to_final_output(output.squeeze(1))

                scores, indices = torch.topk(F.log_softmax(output_logits, dim=1), beam_width, dim=1)

                beam_scores = beam_scores.unsqueeze(2) + scores.unsqueeze(1)

                if t == 0:
                    beam_scores = beam_scores.squeeze(1)
                    decoder_hidden = decoder_hidden.repeat(1, beam_width, 1)
                else:
                    beam_scores = beam_scores.view(batch_size, -1)

                topk_scores, topk_beam_indices = torch.topk(beam_scores, beam_width, dim=1)

                prev_beam_indices = topk_beam_indices // beam_width
                next_input_indices = topk_beam_indices % beam_width

                decoder_hidden = decoder_hidden.view(batch_size, beam_width, -1)
                decoder_hidden = decoder_hidden.gather(1, prev_beam_indices.unsqueeze(2).repeat(1, 1, decoder_hidden.size(2)))

                beam_seq = beam_seq.gather(1, prev_beam_indices.unsqueeze(2).repeat(1, 1, beam_seq.size(2)))
                beam_seq[:, :, t + 1] = indices.view(batch_size, beam_width)[torch.arange(batch_size).unsqueeze(1), prev_beam_indices]

                decoder_input = indices.view(batch_size, beam_width)[torch.arange(batch_size), next_input_indices].unsqueeze(1)

                decoder_hidden = decoder_hidden.view(batch_size * beam_width, -1)
                beam_scores = topk_scores.view(batch_size * beam_width)

            _, best_beam_indices = torch.max(beam_scores.view(batch_size, beam_width), dim=1)
            best_seqs = beam_seq[torch.arange(batch_size), best_beam_indices]

            return best_seqs



    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85)) # inverse of the logit function

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feat, mode, beam_width = 5,target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state, encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        elif mode == 'beam_search':
            seq_logProb, seq_predictions = self.decoder.beam_search(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions

import numpy as np
import torch
import pdb
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from itertools import ifilter

class SquadModel(nn.Module):
    # check if this vocab_size contains unkown, START, END token as well?
    def __init__(self, config, emb_data):
        super(SquadModel, self).__init__()
        # an embedding layer to lookup pre-trained word embeddings
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.embed.weight.requires_grad = False  # do not propagate into the pre-trained word embeddings
        self.embed.weight.data.copy_(emb_data)

        # used for eq(6) does FFNN(p_i)*FFNN(q_j)
        self.ff_align = nn.Linear(config.emb_dim, config.ff_dim)

        # used for eq(2) does FFNN(h_a) in a simplified form so that it can be re-used,
        # note: h_a = [u,v] where u and v are start and end words respectively
        # we have 2*config.hidden_dim since we are using a bi-directional LSTM
        self.p_end_ff = nn.Linear(2 * config.hidden_dim, config.ff_dim)
        self.p_start_ff = nn.Linear(2 * config.hidden_dim, config.ff_dim)

        # used for eq(2) plays the role of w_a
        self.w_a = nn.Linear(config.ff_dim, 1, bias=False)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.2)
        self.hidden = self.init_hidden(config.num_layers, config.hidden_dim, config.batch_size)
        # since we are using q_align and p_emb as p_star we have input as 2*emb_dim
        # num_layers = 2 and dropout = 0.1
        self.gru = nn.GRU(input_size = 2 * config.emb_dim, hidden_size = config.hidden_dim, num_layers = config.num_layers, dropout=0.1, bidirectional=True)
        #change init_hidden when you change this gru/lstm

        parameters = ifilter(lambda p: p.requires_grad, self.parameters())
        for p in parameters:
            self.init_param(p)

    def forward(self, config, p, p_mask, p_lens, q, q_mask, q_lens):
        #all these inputs are of type autograd Variable
        max_p_len, max_q_len = p_lens.data.max(), q_lens.data.max() #the output is singleton scalars
        # p is (max_p_len, batch_size) so is p_mask
        # q is (max_q_len, batch_size) so is q_mask

        # pytroch embedding layer contract
        # Input: LongTensor (N, W), N = mini-batch, W = number of indices to extract per mini-batch
        # Output: (N, W, embedding_dim)
        #pdb.set_trace()
        p_emb = self.embed(p.transpose(0, 1))
        p_emb = p_emb.permute(1, 0, 2)
        q_emb = self.embed(q.transpose(0, 1))
        q_emb = q_emb.permute(1, 0, 2)

        # p_emb (max_p_len, batch_size, emb_dim)
        # q_emb (max_q_len, batch_size, emb_dim)

        p_star_parts = [p_emb]  # its a list,later we concatenate them into a tensor/variable
        p_star_dim = config.emb_dim

        #basiclly bring down the p_emb to ff_dim from embedding_dim, apply ReLU to it as well
        q_align_ff_p = self.sequence_linear_layer(self.ff_align, p_emb)  # (max_p_len, batch_size, ff_dim)
        q_align_ff_q = self.sequence_linear_layer(self.ff_align, q_emb)  # (max_q_len, batch_size, ff_dim)

        #randomly with probability 0.2 zero out some nodes
        q_align_ff_p = self.dropout(q_align_ff_p)
        q_align_ff_q = self.dropout(q_align_ff_q)
        
        q_align_ff_p_shuffled = q_align_ff_p.permute(1, 0, 2)  # (batch_size, max_p_len, ff_dim)
        q_align_ff_q_shuffled = q_align_ff_q.permute(1, 2, 0)  # (batch_size, ff_dim, max_q_len)

        q_align_scores = torch.bmm(q_align_ff_p_shuffled, q_align_ff_q_shuffled)  # (batch_size, max_p_len, max_q_len)
        #q_align_scores is basiaclly s_ij from the paper equation(6) i indexes paragraph, j indexes question

        # p_mask has dimensions (max_p_len, batch_size)
        p_mask_shuffled = torch.unsqueeze(p_mask, 2)  # results in (max_p_len, batch_size, 1)
        p_mask_shuffled = p_mask_shuffled.permute(1, 0, 2)  # (batch_size, max_p_len, 1)

        q_mask_shuffled = torch.unsqueeze(q_mask, 2)  # results in (max_q_len, batch_size, 1)
        q_mask_shuffled = q_mask_shuffled.permute(1, 2, 0)  # (batch_size, 1, max_q_len)
        pq_mask = torch.bmm(p_mask_shuffled, q_mask_shuffled)  # (batch_size, max_p_len, max_q_len)
        pq_mask = pq_mask.float()
        q_align_mask_scores = q_align_scores * pq_mask  # elementwise matrix multiplication
        #not all (i,j) pairs of indexes are valid so we mask the invalid ones out
        q_align_weights = self.sequence_softmax(q_align_mask_scores)  #softmax across the q_len index=2, this is what we do in eqn(7)
        q_emb_shuffled = q_emb.permute(1, 0, 2)  # (batch_size, max_q_len, emb_dim)

        q_align = torch.bmm(q_align_weights, q_emb_shuffled)  # (batch_size, max_p_len, emb_dim)
        q_align_shuffled = q_align.permute(1, 0, 2)  # (max_p_len, batch_size, emd_dim)

        p_star_parts.append(q_align_shuffled)
        p_star_dim += config.emb_dim

        p_star = torch.cat(p_star_parts, 2)  # (max_p_len, batch_size, p_star_dim)
        p_level_h, self.hidden = self.gru(p_star, self.hidden)  # (max_p_len, batch_size, 2*hidden_dim)

        p_stt_lin = self.sequence_linear_layer(self.p_start_ff, p_level_h)  # (max_p_len, batch_size, ff_dim)
        p_end_lin = self.sequence_linear_layer(self.p_end_ff, p_level_h)  # (max_p_len, batch_size, ff_dim)

        # (batch_size, max_p_len*max_ans_len, ff_dim), (batch_size, max_p_len*max_ans_len)
        span_lin_reshaped, span_masks_reshaped = self._span_sums(p_lens, p_stt_lin, p_end_lin, max_p_len, config.batch_size, config.ff_dim, config.max_ans_len)
        #FFNN(h_a) also contains a relu so we are apply that to the whole span_sum
        span_ff_reshaped = self.relu(span_lin_reshaped)  # (batch_size, max_p_len*max_ans_len, ff_dim)
        
        span_scores_reshaped = self.sequence_linear_layer2(self.w_a, span_ff_reshaped)  # (batch_size, max_p_len*max_ans_len)
        final_span_scores = span_masks_reshaped * span_scores_reshaped
        return self.logsoftmax(final_span_scores)

    #input has dimension (batch_size, max_p_len*ans_len, ff_dim)
    def sequence_linear_layer2(self, layer, inp):
        bat_size, classes, inp_dim = inp.size()
        inp = inp.contiguous().view(-1, inp_dim)
        out = layer(inp)
        out = out.view(bat_size, classes, -1)
        return out

    # input has dimension (sequence_len, batch_size, input_dim)
    # ayush's idea to optimize this is to use a view, data is read/unread rowise
    def sequence_linear_layer(self, layer, inp):
        seq_len, bat_size, inp_dim = inp.size()
        #contiguous is called to make the backend memory storage of the tensor as a contigous storage
        # instead of storage with holes
        inp = inp.contiguous().view(-1, inp_dim)
        out = self.relu(layer(inp))
        out = out.view(seq_len, bat_size, -1)
        return out

    def init_hidden(self, num_layers, hidden_dim, batch_size):
        """
		h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
		"""
        return (Variable(torch.zeros(num_layers * 2, batch_size, hidden_dim)))#, Variable(torch.zeros(num_layers * 2, batch_size, hidden_dim)))

    def init_param(self, param):
        if len(param.size()) < 2:
            init.uniform(param)
        else:            
            init.xavier_uniform(param)

    def _span_sums(self, p_lens, stt, end, max_p_len, batch_size, dim, max_ans_len):
        # stt 		(max_p_len, batch_size, dim)
        # end 		(max_p_len, batch_size, dim)
        # p_lens 	(batch_size,)

        max_ans_len_range = torch.from_numpy(np.arange(max_ans_len))
        max_ans_len_range = max_ans_len_range.unsqueeze(0)  # (1, max_ans_len) is a vector like [0,1,2,3,4....,max_ans_len-1]
        offsets = torch.from_numpy(np.arange(max_p_len))
        offsets = offsets.unsqueeze(0)  # (1, max_p_len) is a vector like (0,1,2,3,4....max_p_len-1)
        offsets = offsets.transpose(0, 1)  # (max_p_len, 1) is row vector now like [0/1/2/3...max_p_len-1]

        end_idxs = max_ans_len_range.expand(offsets.size(0), max_ans_len_range.size(1)) + offsets.expand(offsets.size(0), max_ans_len_range.size(1))
        #pdb.set_trace()
        end_idxs_flat = end_idxs.view(-1, 1).squeeze(1)  # (max_p_len*max_ans_len, )
        # note: this is not modeled as tensor of size (SZ, 1) but vector of SZ size

        end_padded = torch.cat((end, Variable(torch.zeros(max_ans_len - 1, batch_size, dim))), 0)
        end_structed = end_padded[end_idxs_flat]  # (max_p_len*max_ans_len, batch_size, dim)
        end_structed = end_structed.view(max_p_len, max_ans_len, batch_size, dim)
        stt_shuffled = stt.unsqueeze(1)  # stt (max_p_len, 1, batch_size, dim)

        # since the FFNN(h_a) * W we expand h_a as [p_start, p_end]*[w_1 w_2] so this reduces to p_start*w_1 + p_end*w_2
        # now we can reuse the operations, we compute only once
        span_sums = stt_shuffled.expand(max_p_len, max_ans_len, batch_size, dim) + end_structed # (max_p_len, max_ans_len, batch_size, dim)
        
        span_sums_reshapped = span_sums.permute(2, 0, 1, 3).contiguous().view(batch_size, max_ans_len * max_p_len, dim)

        p_lens_shuffled = p_lens.unsqueeze(1)
        end_idxs_flat_shuffled = end_idxs_flat.unsqueeze(0)

        span_masks_reshaped = Variable(end_idxs_flat_shuffled.expand(p_lens_shuffled.size(0), end_idxs_flat_shuffled.size(1))) < p_lens_shuffled.expand(p_lens_shuffled.size(0), end_idxs_flat_shuffled.size(1))
        span_masks_reshaped = span_masks_reshaped.float()

        return span_sums_reshapped, span_masks_reshaped

    #q_align_weights = self.softmax(q_align_mask_scores)  # (batch_size, max_p_len, max_q_len)
    def sequence_softmax(self, mat):
        nmat = Variable(torch.randn(mat.size()))
        for i in range(mat.size(1)):
            nmat[:,i,:] = self.softmax(mat[:,i,:])
        return nmat
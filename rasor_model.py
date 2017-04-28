import numpy as np
import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable

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
        self.logsoftmax = nn.LogSoftmax()

        self.hidden = self.init_hidden(config.num_layers, config.hidden_dim, config.batch_size)
        # since we are using q_align and p_emb as p_star we have input as 2*emb_dim
        # num_layers = 2 and dropout = 0.1
        self.gru = nn.GRU(2 * config.emb_dim, config.hidden_dim, config.num_layers, 0.1, bidirectional=True)
        self.cross_ents = nn.CrossEntropyLoss()

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

        q_align_ff_p = self.sequence_linear_layer(self.ff_align, p_emb)  # (max_p_len, batch_size, ff_dim)
        q_align_ff_q = self.sequence_linear_layer(self.ff_align, q_emb)  # (max_q_len, batch_size, ff_dim)

        q_align_ff_p_shuffled = q_align_ff_p.permute(1, 0, 2)  # (batch_size, max_p_len, ff_dim)
        q_align_ff_q_shuffled = q_align_ff_q.permute(1, 2, 0)  # (batch_size, ff_dim, max_q_len)

        q_align_scores = torch.bmm(q_align_ff_p_shuffled, q_align_ff_q_shuffled)  # (batch_size, max_p_len, max_q_len)
        
        # p_mask has dimensions (max_p_len, batch_size)
        p_mask_shuffled = torch.unsqueeze(p_mask, 2)  # results in (max_p_len, batch_size, 1)
        p_mask_shuffled = p_mask_shuffled.permute(1, 0, 2)  # (batch_size, max_p_len, 1)

        q_mask_shuffled = torch.unsqueeze(q_mask, 2)  # results in (max_q_len, batch_size, 1)
        q_mask_shuffled = q_mask_shuffled.permute(1, 2, 0)  # (batch_size, 1, max_q_len)
        pq_mask = torch.bmm(p_mask_shuffled, q_mask_shuffled)  # (batch_size, max_p_len, max_q_len)
        pq_mask = pq_mask.float()
        q_align_mask_scores = q_align_scores * pq_mask  # elementwise matrix multiplication

        # this internal pytorch softmax automatically does max, min shifting to prevent overflows
        #q_align_weights = self.softmax(q_align_mask_scores)  # (batch_size, max_p_len, max_q_len)
        q_align_weights = self.softmax_depths_with_mask(q_align_scores, pq_mask)
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

        span_ff_reshaped = self.relu(span_lin_reshaped)  # (batch_size, max_p_len*max_ans_len, ff_dim)
        span_scores_reshaped = self.sequence_linear_layer(self.w_a, span_ff_reshaped)  # (batch_size, max_p_len*max_ans_len)

        final_span_scores = span_masks_reshaped * span_scores_reshaped
        return self.logsoftmax(final_span_scores)

    # input has dimension (sequence_len, batch_size, input_dim)
    def sequence_linear_layer(self, layer, inp):
        dims = inp.size()
        out = []
        for i in range(0,dims[0]):
            inp_i = inp[i, :, :]
            out_i = self.relu(layer(inp_i))
            out.append(out_i)
        return torch.stack(out, 0)

    def init_hidden(self, num_layers, hidden_dim, batch_size):
        """
		h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
		"""
        return Variable(torch.zeros(num_layers * 2, batch_size, hidden_dim))

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

    def softmax_depths_with_mask(self, x, mask):
        #x has the dimension of (batch_size, max_p_len, max_q_len)
        #mask has the dimension of (batch_size, max_p_len, max_q_len)    
        x = x * mask
        x_min, _ = x.min(2)
        x_min = x_min.expand(x.size(0), x.size(1), x.size(2))
        x -= x_min
        x = x * mask
        x_max, _ = x.max(2)
        x_max = x_max.expand(x.size(0), x.size(1), x.size(2))
        x -= x_max
        e_x = mask * torch.exp(x)
        sums = e_x.sum(2)
        denom = sums + torch.eq(sums, 0).float()
        denom = denom.expand(denom.size(0), denom.size(1), e_x.size(2))
        y = e_x / denom
        y = y * mask
        return y

from torch import optim
import torch.nn as nn
from rasor_model import SquadModel
from itertools import ifilter
import sys
import logging, pdb
import time
import os.path
from torch.autograd import Variable
import argparse
import torch
import numpy as np
from evaluate11 import metric_max_over_ground_truths, exact_match_score, f1_score
from base.utils import set_up_logger
from utils import EpochResult, format_epoch_results, plot_epoch_results
from reader import get_data, construct_answer_hat, write_test_predictions
import subprocess
import signal

class Config(object):
    def __init__(self, compared=[], **kwargs):
        self.name = "RaSoR"
        self.word_emb_data_path_prefix = 'data/preprocessed_glove_with_unks.split'  # path of preprocessed word embedding data, produced by setup.py
        self.tokenized_trn_json_path = 'data/train-v1.1.tokenized.split.json'  # path of tokenized training set JSON, produced by setup.py
        self.tokenized_dev_json_path = 'data/dev-v1.1.tokenized.split.json'  # path of tokenized dev set JSON, produced by setup.py
        self.max_ans_len = 30  # maximal answer length, answers of longer length are discarded
        self.emb_dim = 300  # dimension of word embeddings
        self.ff_dim = 100
        self.batch_size = 40
        self.max_num_epochs = 150  # max number of epochs to train for
        self.num_layers = 2 # number of BiLSTM layers, where BiLSTM is applied
        self.hidden_dim = 100  # dimension of hidden state of each uni-directional LSTM
        self.vocab_size = 114885
    
    def __repr__(self):
        ks = sorted(k for k in self.__dict__ if k not in ['name'])
        return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)

def _gpu_dataset(name, dataset, config):
    if dataset:
        ds_vec = dataset.vectorized
        ctxs, ctx_masks, ctx_lens = _gpu_sequences(name + '_ctxs', ds_vec.ctxs, ds_vec.ctx_lens)
        qtns, qtn_masks, qtn_lens = _gpu_sequences(name + '_qtns', ds_vec.qtns, ds_vec.qtn_lens)
        qtn_ctx_idxs = torch.from_numpy(ds_vec.qtn_ctx_idxs)
        anss, ans_stts, ans_ends = _gpu_answers(name, ds_vec.anss, config.max_ans_len)
    else:
        ctxs = ctx_masks = qtns = qtn_masks = torch.zeros(1, 1)
        ctx_lens = qtn_lens = qtn_ctx_idxs = anss = ans_stts = ans_ends = torch.zeros(1)
    return ctxs, ctx_masks, ctx_lens, qtns, qtn_masks, qtn_lens, qtn_ctx_idxs, anss, ans_stts, ans_ends

def _gpu_sequences(name, seqs_val, lens):
    assert seqs_val.dtype == lens.dtype == np.int32
    num_samples, max_seq_len = seqs_val.shape
    assert len(lens) == num_samples
    assert max(lens) == max_seq_len
    gpu_seqs = torch.from_numpy(seqs_val)
    seq_masks_val = np.zeros((num_samples, max_seq_len), dtype=np.int32)
    for i, sample_len in enumerate(lens):
        seq_masks_val[i, :sample_len] = 1
        assert np.all(seqs_val[i, :sample_len] > 0)
        assert np.all(seqs_val[i, sample_len:] == 0)
    gpu_seq_masks = torch.from_numpy(seq_masks_val)
    gpu_lens = torch.from_numpy(lens)
    return gpu_seqs, gpu_seq_masks, gpu_lens

def _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len):
    # all arguments are concrete ints
    assert ans_end_word_idx - ans_start_word_idx + 1 <= max_ans_len
    return ans_start_word_idx * max_ans_len + (ans_end_word_idx - ans_start_word_idx)

def _tt_ans_idx_to_ans_word_idxs(ans_idx, max_ans_len):
    # ans_idx theano int32 variable (batch_size,)
    # max_ans_len concrete int
    ans_start_word_idx = ans_idx // max_ans_len
    ans_end_word_idx = ans_start_word_idx + ans_idx % max_ans_len
    return ans_start_word_idx, ans_end_word_idx

def _gpu_answers(name, anss, max_ans_len):
    assert anss.dtype == np.int32
    assert anss.shape[1] == 2
    anss_val = np.array([_np_ans_word_idxs_to_ans_idx(ans_stt, ans_end, max_ans_len) for \
                         ans_stt, ans_end in anss], dtype=np.int32)
    ans_stts_val = anss[:, 0]
    ans_ends_val = anss[:, 1]

    gpu_anss = torch.from_numpy(anss_val)
    gpu_ans_stts = torch.from_numpy(ans_stts_val)
    gpu_ans_ends = torch.from_numpy(ans_ends_val)
    return gpu_anss, gpu_ans_stts, gpu_ans_ends

config = Config()
base_filename = config.name + '_cfg' + str(0)
logger = set_up_logger('logs/' + base_filename + '.log')
title = '{}: {}'.format(__file__, config.name)
logger.info('START ' + title + '\n\n{}\n'.format(config))
data = get_data(config, train=True)


emb_val = data.word_emb_data.word_emb  # (voc size, emb_dim)
first_known_word = data.word_emb_data.first_known_word
assert config.emb_dim == emb_val.shape[1]
assert first_known_word > 0
emb_val[:first_known_word] = 0

emb = torch.from_numpy(emb_val)

trn_ctxs, trn_ctx_masks, trn_ctx_lens, trn_qtns, trn_qtn_masks, trn_qtn_lens, trn_qtn_ctx_idxs, trn_anss, trn_ans_stts, trn_ans_ends = _gpu_dataset(
    'trn', data.trn, config)

dataset_ctxs = trn_ctxs.long()
dataset_ctx_masks = trn_ctx_masks.long()
dataset_ctx_lens = trn_ctx_lens.long()
dataset_qtns = trn_qtns.long()
dataset_qtn_masks = trn_qtn_masks.long()
dataset_qtn_lens = trn_qtn_lens.long()
dataset_qtn_ctx_idxs = trn_qtn_ctx_idxs.long()
dataset_anss = trn_anss.long()
dataset_ans_stts = trn_ans_stts.long()
dataset_ans_ends = trn_ans_ends.long()

if torch.cuda.is_available():
    print("gpu pe dauhdha!")
    gpu_avail = torch.cuda.device_count()
    print(gpu_avail)
    dataset_ctxs = trn_ctxs.long().cuda(0)
    dataset_ctx_masks = trn_ctx_masks.long().cuda(0)
    dataset_ctx_lens = trn_ctx_lens.long().cuda(0)
    dataset_qtns = trn_qtns.long().cuda(0)
    dataset_qtn_masks = trn_qtn_masks.long().cuda(0)
    dataset_qtn_lens = trn_qtn_lens.long().cuda(0)
    dataset_qtn_ctx_idxs = trn_qtn_ctx_idxs.long().cuda(0)
    dataset_anss = trn_anss.long().cuda(0)
    dataset_ans_stts = trn_ans_stts.long().cuda(0)
    dataset_ans_ends = trn_ans_ends.long().cuda(0)


def print_param(mdoel):
    for name, param in model.state_dict().items():
        print(name,)

    print('\n')
    for param in list(model.parameters()):
        print(param.data.size())

loss_function = nn.NLLLoss()

# indices of questions which have a valid answer
valid_qtn_idxs = np.flatnonzero(data.trn.vectorized.qtn_ans_inds).astype(np.int32)
num_samples = valid_qtn_idxs.size
# probably shuffle the sample each epoch
batch_sizes = []
losses = []
accs = []

def clean():
    with open("meta_data.txt") as fp:
        avg_trn_loss = np.average(losses)
        avg_trn_acc = np.average(accs)
        fp.write("avg_loss: {} avg_acc: {} \n".format(avg_trn_loss, avg_trn_acc))
        torch.save(model.state_dict(), './model.pth')

signal.signal(signal.SIGTERM, clean)

def _trn_epoch(model, epochid, batchid = 0):
    lr = 0.001
    #TODO: change this  to iterate over all samples, now we limit to 25% of the sample size
    ss = range(0, num_samples, config.batch_size)
    for b, s in enumerate(ss, 1):
        if b < batchid:
            #skip all batchids less than the given batchid
            continue

        batch_idxs = valid_qtn_idxs[s:min(s + config.batch_size, num_samples)]
        if batch_idxs.size != config.batch_size:
            #in the last iteration if the size of the vector is not as batch size
            #GRU and LSTM layer would fail, since hidden state is initialized with fixed batch_size
            continue

        qtn_idxs = torch.from_numpy(batch_idxs).long()
        ctx_idxs = dataset_qtn_ctx_idxs[qtn_idxs].long()  # (batch_size,)
        p_lens = dataset_ctx_lens[ctx_idxs]  # (batch_size,)
        #pdb.set_trace()
        max_p_len = p_lens.max()
        p = dataset_ctxs[ctx_idxs][:, :max_p_len].transpose(0, 1)  # (max_p_len, batch_size)
        p_mask = dataset_ctx_masks[ctx_idxs][:, :max_p_len].transpose(0, 1).long()  # (max_p_len, batch_size)
        float_p_mask = p_mask.float()

        q_lens = dataset_qtn_lens[qtn_idxs]  # (batch_size,)
        max_q_len = q_lens.max()
        q = dataset_qtns[qtn_idxs][:, :max_q_len].transpose(0, 1)  # (max_q_len, batch_size)
        q_mask = dataset_qtn_masks[qtn_idxs][:, :max_q_len].transpose(0, 1).long()  # (max_q_len, batch_size)
        float_q_mask = q_mask.float()

        a = Variable(dataset_anss[qtn_idxs])  # (batch_size,)
        #a_stt = dataset_ans_stts[qtn_idxs]  # (batch_size,)
        #a_end = dataset_ans_ends[qtn_idxs]  # (batch_size,)

        batch_sizes.append(len(batch_idxs))

        start_time = time.time()
        
        model.zero_grad()
        model.hidden = model.init_hidden(config.num_layers, config.hidden_dim, config.batch_size)

        scores = model(config, Variable(p, requires_grad=False), Variable(p_mask, requires_grad=False),
                       Variable(p_lens, requires_grad=False), Variable(q, requires_grad=False),
                       Variable(q_mask, requires_grad=False), Variable(q_lens, requires_grad=False))

        parameters = ifilter(lambda p: p.requires_grad, model.parameters())
        #pdb.set_trace()

        optimizer = optim.Adam(parameters, lr)
        loss = loss_function(scores, a)
        
        _, a_hats = torch.max(scores, 1)
        a_hats = a_hats.squeeze(1)
        acc = torch.eq(a_hats, a).float().mean()
        
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        accs.append(acc.data[0])
        
        if b % 1100 == 0:
            #implementing a learning rate decay
            lr = lr * 0.95
            parameters = ifilter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.Adam(parameters, lr)

        if b % 10 == 0:
            logger.info("loss: {} accuracy:{} epochID: {} batchID:{}".format(loss.data[0], acc.data[0], epochid, b))

        if b % 100 == 0:
            #save frequency, creates a 140MB file
            torch.save(model.state_dict(), './model.pth')

    trn_loss = np.average(losses)
    trn_acc = np.average(accs)
    return trn_loss, trn_acc

def main():
    model = SquadModel(config, emb)
    #check for old model if present
    if os.path.isfile('./model.pth'):
        model.load_state_dict(torch.load('./model.pth'))
        #load the model from here instead 
        batchid = get_last_batch() + 1
    
    #Training
    for epoch in range(20):
        trn_loss, trn_acc = _trn_epoch(model, epoch, batchid)
        batchid = 0
        logger.info("after epoch: {} avg. loss: {} avg. acc: {} ".format(epoch, trn_loss, trn_acc))

def get_last_batch():
    #since its possible that the last line was partly pasted to stdout
    proc = subprocess.Popen(['tail', '-2', 'program_round1.out'], stdout=subprocess.PIPE)
    val = proc.stdout.read().split('\n')[0]
    batchid = int(val.strip().split(':')[-1])
    return batchid

main()
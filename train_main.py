from torch import optim
import torch.nn as nn
from rasor_model import SquadModel
from itertools import ifilter
import sys
import logging, pdb
import time
from torch.autograd import Variable
import argparse
import torch
import numpy as np
from evaluate11 import metric_max_over_ground_truths, exact_match_score, f1_score
from base.utils import set_up_logger
from utils import EpochResult, format_epoch_results, plot_epoch_results
from reader import get_data, construct_answer_hat, write_test_predictions


class Config(object):
    def __init__(self, compared=[], **kwargs):
        self.name = None
        self.desc = None
        self.device = None  # 'cpu' / 'gpu<index>'
        self.plot = False  # whether to plot training graphs
        self.save_freq = None  # how often to save model (in epochs); None for only after best EM/F1 epochs

        self.word_emb_data_path_prefix = None  # path of preprocessed word embedding data, produced by setup.py
        self.tokenized_trn_json_path = None  # path of tokenized training set JSON, produced by setup.py
        self.tokenized_dev_json_path = None  # path of tokenized dev set JSON, produced by setup.py
        self.test_json_path = None  # path of test set JSON
        self.pred_json_path = None  # path of test predictions JSON
        self.tst_load_model_path = None  # path of trained model data, used for producing test set predictions
        self.tst_split = True  # whether to split hyphenated unknown words of test set, see setup.py

        self.seed = np.random.random_integers(1e6, 1e9)
        self.max_ans_len = 30  # maximal answer length, answers of longer length are discarded
        self.emb_dim = 300  # dimension of word embeddings
        self.learn_single_unk = False  # whether to have a single tunable word embedding for all unknown words
        # (or multiple fixed random ones)
        self.init_scale = 5e-3  # uniformly random weights are initialized in [-init_scale, +init_scale]
        self.learning_rate = 1e-3
        self.lr_decay = 0.95
        self.lr_decay_freq = 5000  # frequency with which to decay learning rate, measured in updates
        self.max_grad_norm = 10  # gradient clipping
        self.ff_dims = [100]  # dimensions of hidden FF layers
        self.ff_dim = 100
        self.ff_drop_x = 0.2  # dropout rate of FF layers
        self.batch_size = 40
        self.max_num_epochs = 150  # max number of epochs to train for

        self.num_bilstm_layers = 2  # number of BiLSTM layers, where BiLSTM is applied
        self.num_layers = 2
        self.hidden_dim = 100  # dimension of hidden state of each uni-directional LSTM
        self.lstm_drop_h = 0.1  # dropout rate for recurrent hidden state of LSTM
        self.lstm_drop_x = 0.4  # dropout rate for inputs of LSTM
        self.lstm_couple_i_and_f = True  # customizable LSTM configuration, see base/model.py
        self.lstm_learn_initial_state = False
        self.lstm_tie_x_dropout = True
        self.lstm_sep_x_dropout = False
        self.lstm_sep_h_dropout = False
        self.lstm_w_init = 'uniform'
        self.lstm_u_init = 'uniform'
        self.lstm_forget_bias_init = 'uniform'
        self.default_bias_init = 'uniform'

        self.extra_drop_x = 0  # dropout rate at an extra possible place
        self.q_aln_ff_tie = True  # whether to tie the weights of the FF over question and the FF over passage
        self.sep_stt_end_drop = True  # whether t302o have separate dropout masks for span start and
        # span end representations

        self.adam_beta1 = 0.9  # see base/optimizer.py
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8

        self.objective = 'span_multinomial'  # 'span_multinomial': multinomial distribution over all spans
        # 'span_binary':      logistic distribution per span
        # 'span_endpoints':   two multinomial distributions, over span start and end

        self.ablation = 'only_q_align'  # 'only_q_align':     question encoded only by passage-aligned representation
        # 'only_q_indep':     question encoded only by passage-independent representation
        # None:               question encoded by both
        self.vocab_size = 114885
        assert all(k in self.__dict__ for k in kwargs)
        assert all(k in self.__dict__ for k in compared)
        self.__dict__.update(kwargs)
        self._compared = compared


def _get_configs():
    compared = [
        'device', 'objective', 'ablation', 'batch_size', 'ff_dims', 'ff_drop_x',
        'hidden_dim', 'lstm_drop_h', 'lstm_drop_x', 'lstm_drop_h', 'q_aln_ff_tie']
    common = {
        'name': 'RaSoR',
        'desc': 'Recurrent span representations',
        'word_emb_data_path_prefix': 'data/preprocessed_glove_with_unks.split',
        'tokenized_trn_json_path': 'data/train-v1.1.tokenized.split.json',
        'tokenized_dev_json_path': 'data/dev-v1.1.tokenized.split.json',
        'plot': True
    }
    configs = [

        # Objective comparison:

        Config(compared,
               objective='span_multinomial',
               tst_load_model_path='models/RaSoR_cfg0_best_em.pkl',
               **common),

        Config(compared,
               objective='span_binary',
               tst_load_model_path='models/RaSoR_cfg1_best_em.pkl',
               **common),

        Config(compared,
               objective='span_endpoints',
               tst_load_model_path='models/RaSoR_cfg2_best_em.pkl',
               **common),

        # Ablation study:

        Config(compared,
               objective='span_multinomial',
               ablation='only_q_align',
               tst_load_model_path='models/RaSoR_cfg3_best_em.pkl',
               **common),

        Config(compared,
               objective='span_multinomial',
               ablation='only_q_indep',
               tst_load_model_path='models/RaSoR_cfg4_best_em.pkl',
               **common),

    ]

    return configs


parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device e.g. cpu, gpu0, gpu1, ...', default='cpu')
parser.add_argument('--train', help='whether to train', action='store_true')
parser.add_argument('--cfg_idx', help='configuration index', type=int, default=0)
parser.add_argument('test_json_path', nargs='?', help='test JSON file for which answers should be predicted')
parser.add_argument('pred_json_path', nargs='?', help='where to write test predictions to')
args = parser.parse_args()
if bool(args.test_json_path) != bool(args.pred_json_path) or bool(args.test_json_path) == args.train:
    parser.error('Specify both test_json_path and pred_json_path, or only --train')
config = _get_configs()[args.cfg_idx]
config.device = args.device
config.test_json_path = args.test_json_path
config.pred_json_path = args.pred_json_path

config_idx = args.cfg_idx
train = args.train

base_filename = config.name + '_cfg' + str(config_idx)
logger = set_up_logger('logs/' + base_filename + '.log')
title = '{}: {} ({}) config index {}'.format(__file__, config.name, config.desc, config_idx)
logger.info('START ' + title + '\n\n{}\n'.format(config))

data = get_data(config, train)


# print(type(data[1].vectorized.ctxs))


##################################################
# Variable-length data to GPU matrices and masks
###################################################

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


emb_val = data.word_emb_data.word_emb  # (voc size, emb_dim)
first_known_word = data.word_emb_data.first_known_word
assert config.emb_dim == emb_val.shape[1]
assert first_known_word > 0
emb_val[:first_known_word] = 0

emb = torch.from_numpy(emb_val)

trn_ctxs, trn_ctx_masks, trn_ctx_lens, trn_qtns, trn_qtn_masks, trn_qtn_lens, trn_qtn_ctx_idxs, trn_anss, trn_ans_stts, trn_ans_ends = _gpu_dataset(
    'trn', data.trn, config)

# dev_ctxs, dev_ctx_masks, dev_ctx_lens, dev_qtns, dev_qtn_masks, dev_qtn_lens, dev_qtn_ctx_idxs,dev_anss, dev_ans_stts, dev_ans_ends = _gpu_dataset('dev', data.dev, config)

# tst_ctxs, tst_ctx_masks, tst_ctx_lens, tst_qtns, tst_qtn_masks, tst_qtn_lens, tst_qtn_ctx_idxs,tst_anss, tst_ans_stts, tst_ans_ends = _gpu_dataset('tst', data.tst, config)

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

model = SquadModel(config, emb)

loss_function = nn.CrossEntropyLoss()

def _trn_epoch():
    logger = logging.getLogger()
    # indices of questions which have a valid answer
    valid_qtn_idxs = np.flatnonzero(data.trn.vectorized.qtn_ans_inds).astype(np.int32)
    # todo shuffle in numpy np_rng.shuffle(valid_qtn_idxs)
    num_samples = valid_qtn_idxs.size
    batch_sizes = []
    losses = []
    accs = []
    samples_per_sec = []
    ss = range(0, num_samples, config.batch_size)
    for b, s in enumerate(ss, 1):

        batch_idxs = valid_qtn_idxs[s:min(s + config.batch_size, num_samples)]
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

        a_hats = model(config, Variable(p, requires_grad=False), Variable(p_mask, requires_grad=False),
                       Variable(p_lens, requires_grad=False), Variable(q, requires_grad=False),
                       Variable(q_mask, requires_grad=False), Variable(q_lens, requires_grad=False))

        parameters = ifilter(lambda p: p.requires_grad, model.parameters())
        #pdb.set_trace()
        optimizer = optim.Adam(parameters)
        loss = loss_function(a_hats, a)
        loss.backward()
        optimizer.step()

        samples_per_sec.append(len(batch_idxs) / (time.time() - start_time))

        losses.append(loss)
        print("loss", loss.data[0])

    trn_loss = np.average(losses, weights=batch_sizes)
    #trn_acc = np.average(accs, weights=batch_sizes)
    trn_samples_per_sec = np.average(samples_per_sec, weights=batch_sizes)
    return trn_loss, trn_samples_per_sec


_trn_epoch()

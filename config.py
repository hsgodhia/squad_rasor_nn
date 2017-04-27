class Config(object):
  def __init__(self, compared=[], **kwargs):
	self.name = None
	self.desc = None
	self.device = None                      # 'cpu' / 'gpu<index>'
	self.plot = False                       # whether to plot training graphs
	self.save_freq = None                   # how often to save model (in epochs); None for only after best EM/F1 epochs

	self.word_emb_data_path_prefix = None   # path of preprocessed word embedding data, produced by setup.py
	self.tokenized_trn_json_path = None     # path of tokenized training set JSON, produced by setup.py
	self.tokenized_dev_json_path = None     # path of tokenized dev set JSON, produced by setup.py
	self.test_json_path = None              # path of test set JSON
	self.pred_json_path = None              # path of test predictions JSON
	self.tst_load_model_path = None         # path of trained model data, used for producing test set predictions
	self.tst_split = True                   # whether to split hyphenated unknown words of test set, see setup.py

	self.max_ans_len = 30                   # maximal answer length, answers of longer length are discarded
	self.emb_dim = 300                      # dimension of word embeddings

	self.ff_dims = 100                      # dimensions of hidden FF layers
	self.batch_size = 100
	self.max_num_epochs = 20               # max number of epochs to train for

	self.num_bilstm_layers = 1              # number of BiLSTM layers, where BiLSTM is applied
	self.hidden_dim = 100                   # dimension of hidden state of each uni-directional LSTM
	# -- aded some new configs
	self.vocab_size = 1000

  def __repr__(self):
	ks = sorted(k for k in self.__dict__ if k not in ['name', 'desc', '_compared'])
	return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)

  def format_compared(self):
	return '\n'.join([
	  ''.join('{:12s} '.format(k[:12]) for k in sorted(self._compared)),
	  ''.join('{:12s} '.format(str(self.__dict__[k])[:12]) for k in sorted(self._compared))])
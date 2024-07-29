from daisy.model.MFRecommender import MF
from daisy.model.FMRecommender import FM
from daisy.model.NFMRecommender import NFM
from daisy.model.NGCFRecommender import NGCF
from daisy.model.EASERecommender import EASE
from daisy.model.SLiMRecommender import SLiM
from daisy.model.VAECFRecommender import VAECF
from daisy.model.NeuMFRecommender import NeuMF
from daisy.model.PopRecommender import MostPop
from daisy.model.KNNCFRecommender import ItemKNNCF
from daisy.model.PureSVDRecommender import PureSVD
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.model.LightGCNRecommender import LightGCN
from daisy.utils.splitter import TestSplitter
from daisy.utils.metrics import calc_ranking_results
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.config import init_logger
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from daisy.utils.utils import ensure_dir, get_ur, get_history_matrix, build_candidates_set, get_inter_matrix

drec_model_enum = {
  'mostpop': MostPop,
  'slim': SLiM,
  'itemknn': ItemKNNCF,
  'puresvd': PureSVD,
  'mf': MF,
  'fm': FM,
  'ngcf': NGCF,
  'neumf': NeuMF,
  'nfm': NFM,
  'multi-vae': VAECF,
  'item2vec': Item2Vec,
  'ease': EASE,
  'lightgcn': LightGCN,
}


drec_base_config = {
  'gpu': '0',                       # default: '0',
  'seed': 2022,                     # default: 2022,
  'reproducibility': True,          # default: True,
  'state': None,                    # default: None,
  'optimization_metric': 'ndcg',    # default: 'ndcg',
  'hyperopt_trail': 30,             # default: 30,
  'tune_testset': False,            # default: False,
  'tune_pack': '{}',                # default: '{}',
  'algo_name': 'ngcf',              # default: 'mf',
  'val_method': 'tsbr',             # default: 'tsbr',
  'test_method': 'tsbr',            # default: 'tsbr',
  'fold_num': 5,                    # default: 1,
  'val_size': 0.1,                  # default: 0.1,
  'test_size': 0.2,                 # default: 0.2,
  'topk': 100,                      # default: 50,
  'cand_num': 100,                  # default: 1000,
  'sample_method': 'uniform',       # default: 'uniform',
  'sample_ratio': 0,                # default: 0,
  'num_ng': 4,                      # default: 4,
  'batch_size': 256,                # default: 256,
  'loss_type': 'BPR',               # default: 'BPR',
  'init_method': 'default',         # default: 'default',
  'optimizer': 'default',           # default: 'default',
  'early_stop': True,               # default: True,
  'data_path': 'data/',             # default: 'data/',
  'res_path': None,                 # default: None,
  # 'dataset': zipfile.split('.')[0], # default: 'ml-100k',
  'prepro': '10filter',             # default: '10filter',
  'level': 'ui',                    # default: 'ui',
  'UID_NAME': 'user',               # default: 'user',
  'IID_NAME': 'item',               # default: 'item',
  'INTER_NAME': 'rating',           # default: 'rating',
  'TID_NAME': 'timestamp',          # default: 'timestamp',
  'binary_inter': True,             # default: True,
  'positive_threshold': None,       # default: None,
  'res_path': './recsys_results',
  'metrics': ['recall', 'mrr', 'ndcg', 'hit', 'precision'],   # default: ['recall', 'mrr', 'ndcg', 'hit', 'precision']
  # 'logger': init_stdout_logger(),
  # 'col_mapper': {colnames.uid: 'user', colnames.iid: 'item', colnames.interid: 'rating', colnames.tsid: 'timestamp'},
  # 'user_num': max(ls_users) + 1,
  # 'item_num': max(ls_items) + 1,
}

drec_model_config_enum = {
  'mostpop': {
  },
  'slim': {
    'elastic': 0.1,             # default: 0.1,
    'alpha': 1.0                # default: 1.0
  },
  'itemknn': {
    'maxk': 40,                 # default: 40,
    'shrink': 100,              # default: 100,
    'similarity': 'cosine',     # default: 'cosine',
    'normalize': True           # default: True
  },
  'puresvd': {
    'factors': 150              # default: 150
  },
  'mf': {
    'factors': 100,             # default: 100,
    'epochs': 20,                # default: 20,  <=============
    'lr': 0.01,                 # default: 0.01,
    'reg_1': 0.001,             # default: 0.001,
    'reg_2': 0.001,             # default: 0.001,
  },
  'fm': {
    'factors': 84,              # default: 84,
    'epochs': 20,                # default: 20,  <=============
    'lr': 0.001,                # default: 0.001,
    'reg_1': 0.001,             # default: 0.001,
    'reg_2': 0.001              # default: 0.001
  },
  'ngcf': {
    'factors': 36,              # default: 36,
    'node_dropout': 0.0,        # default: 0.0,
    'mess_dropout': 0.1,        # default: 0.1,
    'lr': 0.01,                 # default: 0.01,
    'reg_1': 0,                 # default: 0,
    'reg_2': 0,                 # default: 0,
    'epochs': 30,               # default: 30,  <=============
    'hidden_size_list': None    # default: None
  },
  'neumf': {
    'factors': 24,              # default: 24,
    'num_layers': 2,            # default: 2,
    'dropout': 0.5,             # default: 0.5,
    'lr': 0.001,                # default: 0.001,
    'epochs': 30,               # default: 30,  <=============
    'reg_1': 0.001,             # default: 0.001,
    'reg_2': 0.001,             # default: 0.001,
    'model_name': NeuMF,        # default: NeuMF,
    'GMF_model': None,          # default: None,
    'MLP_model': None           # default: None
  },
  'nfm': {
    'factors': 30,              # default: 30,
    'act_function': 'relu',     # default: 'relu',
    'num_layers': 2,            # default: 2,
    'batch_norm': True,         # default: True,
    'dropout': 0.5,             # default: 0.5,
    'epochs': 30,               # default: 30,  <=============
    'lr': 0.001,                # default: 0.001,
    'reg_1': 0,                 # default: 0,
    'reg_2': 0                  # default: 0
  },
  'multi-vae': {
    'mlp_hidden_size': None,    # default: None,
    'epochs': 10,               # default: 10,  <=============
    'dropout': 0.5,             # default: 0.5,
    'lr': 0.001,                # default: 0.001,
    'latent_dim': 128,          # default: 128,
    'total_anneal_steps': 100000,  # default: 100000,
    'anneal_cap': 0.2           # default: 0.2
  },
  'item2vec': {
    'lr': 0.001,                # default: 0.001,
    'epochs': 20,                # default: 20,  <=============
    'factors': 100,             # default: 100,
    'rho': 0.5,                 # default: 0.5,
    'context_window': 2         # default: 2
  },
  'ease': {
    'reg': 200.0                # default: 200.0
  },
  'lightgcn': {
    'factors': 64,              # default: 64,
    'lr': 0.01,                 # default: 0.01,
    'reg_1': 0,                 # default: 0,
    'reg_2': 0,                 # default: 0,
    'epochs': 30,               # default: 30,  <=============
    'num_layers': 2             # default: 2
  },
}
import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['ml-1m', 'mind', 'lastfm', 'mooc']
all_models  = ['colakg']

config['bpr_batch_size']    = args.bpr_batch
config['latent_dim_rec']    = args.recdim
config['lightGCN_n_layers'] = args.layer
config['use_drop_edge']     = args.use_drop_edge
config['keep_prob']         = args.keepprob
config['test_u_batch_size'] = args.testbatch
config['lr']                = args.lr
config['decay']             = args.decay
config['neighbor_k']        = args.neighbor_k
config['dropout_i']         = args.dropout_i
config['dropout_u']         = args.dropout_u
config['dropout_n']         = args.dropout_n
config['ablation']          = args.ablation


GPU    = torch.cuda.is_available()
device = torch.device('cuda' if GPU else 'cpu')
CORES  = multiprocessing.cpu_count() // 2
seed   = args.seed

dataset    = args.dataset
model_name = args.model

if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

item_semantic_emb_file = args.item_semantic_emb_file
user_semantic_emb_file = args.user_semantic_emb_file


TRAIN_epochs = 2000
LOAD         = args.load
RESUME       = args.resume
PATH         = args.path
topks        = eval(args.topks)
tensorboard  = args.tensorboard
comment      = args.comment

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

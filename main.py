import argparse

from fscil_trainer import FSCILTrainer
from utils import *

# MODEL_DIR=None
# MODEL_DIR = 'pretrain/best_model_miniimagenet_5426.pth'
MODEL_DIR = 'pretrain/cifar100_id_5498_bs_256.pth'
# MODEL_DIR = 'pretrain/cub200_id_5621_bs_256.pth'
DATA_DIR = 'D:/LLB/data/'
PROJECT = 'baseline'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=1)  ## for code test set to 1 default: 100
    parser.add_argument('-lr_base', type=float, default=0.0001)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[40, 70])
    parser.add_argument('-step', type=int, default=100)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16.0)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=256)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-start_session', type=int, default=0)
    # parser.add_argument('-sessions', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-use_gpu', type=bool, default=True)

    return parser


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = 'C:/Users/Admin/Desktop/FSCIL/CVAEGAN_FSL/pretrain'
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    if args.use_gpu:
        args.num_gpu = set_gpu(args)
    trainer = FSCILTrainer(args)
    trainer.train()

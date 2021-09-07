import argparse
from types import MethodType

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='bilinear model Args')
    parser.add_argument("--mod",  default='train', required=True, choices=['train','test','generate'])
    parser.add_argument("--dataset", required=True, choices=['vqa2','tdiuc','gqa'])
    parser.add_argument("--model", required=True, choices=['ban','san','mlp'])
    parser.add_argument("--load", default='')
    parser.add_argument("--multiGPU", default=False)
    parser.add_argument("--gpuID")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    return args

class Cfgs:
    def __init__(self):
        super(Cfgs, self).__init__()

        # feature configs
        self.MAX_TOKEN = 14
        self.IMG_FEAT_SIZE = 2048

        # model configs
        self.LAYER = 6
        self.HIDDEN_SIZE = 768
        self.MULTI_HEAD = 8
        self.HIDDEN_SIZE_HEAD = int(self.HIDDEN_SIZE/self.MULTI_HEAD)
        self.FF_SIZE = 512
        self.ATT_OUT_SIZE = 1024
        self.DROPOUT_R = 0.1
        self.gamma = 2
        self.h_mm = 768
        self.h_out = 1
        self.k = 1

        # optimizer configs
        self.LR_BASE = 1e-4
        self.LR_DECAY_R = 0.25
        self.LR_DECAY_LIST = [8,10]

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


C = Cfgs()
args = parse_args()
args_dict = C.parse_to_dict(args)
C.add_args(args_dict) 
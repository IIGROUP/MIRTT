import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int,default=17,help="epoch of training")
    parser.add_argument("--lr",type=float,default=1e-4,help="learning rate during training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=0,help="whether use gpu")
    parser.add_argument("--load",type=str,default='')
    parser.add_argument("--dataset",type=str,default='v7w',choices=['v7w','vqa1','vqa2','tdiuc','gqa'],help="select the type of dataset")

    # Only for Trilinear Transformer
    parser.add_argument("--layers",type=int,default=6, help="number of layers in Trilinear Transformer")
    parser.add_argument("--ff_dim",type=int,default=3072, help="dims of Position Wise Feed Forward layer in Trilinear Transformer")
    parser.add_argument("--dropout",type=float,default=0.1, help="dropout rate in Trilinear Transformer")
    
    # Initial
    parser.add_argument("--batch_size",type=int,default=16,help="batch size during training")
    parser.add_argument("--qst_len",type=int,default=12,help="max length of question")
    parser.add_argument("--ans_len",type=int,default=6,help="max length of answer")
    parser.add_argument("--seed",type=int,default=1919,help="seed of random")
    
    parser.add_argument("--output", type=str, default='test/', help="dir of output")
    parser.add_argument("--save_model", action='store_const', default=False, const=True)
    parser.add_argument("--info", type=str, default='info')
    
    return parser.parse_args()
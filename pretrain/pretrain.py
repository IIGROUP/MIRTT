import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import argparse
import os
from tqdm import tqdm, trange

from model import TTpretrainModel
from data import v7wDataset, VQADataset, TDIUCDataset, trim_collate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int,default=15,help="epoch of training")
    parser.add_argument("--lr",type=float,default=3e-5,help="learning rate during training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=2,help="whether use gpu")
    parser.add_argument("--load",type=str,default='')

    # Only for Trilinear Transformer
    parser.add_argument("--num_layers",type=int,default=6, help="number of layers in Trilinear Transformer")
    parser.add_argument("--ff_dim",type=int,default=3072, help="dims of Position Wise Feed Forward layer in Trilinear Transformer")
    parser.add_argument("--dropout",type=float,default=0.1, help="dropout rate in Trilinear Transformer")
    
    # Initial
    parser.add_argument("--batch_size",type=int,default=32,help="batch size during training")
    parser.add_argument("--qst_len",type=int,default=12,help="max length of question")
    parser.add_argument("--ans_len",type=int,default=6,help="max length of answer")
    parser.add_argument("--seed",type=int,default=1919,help="seed of random")
    
    parser.add_argument("--output", type=str, default='test/', help="dir of output")
    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    model = TTpretrainModel(num_layers=config.num_layers, ff_dim=config.ff_dim, dropout=config.dropout)
    if config.load:
        print('Loading ckpt {}'.format(config.load))
        ckpt = torch.load(config.load)
        model.load_state_dict(ckpt)
        print('Finish!')
    model = model.cuda()
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    max_boxes = 50
    max_length = [config.qst_len, config.ans_len]
    v7w_train_dset = v7wDataset('train', adaptive=True, max_boxes=max_boxes, maxlen=max_length)
    v7w_val_dset = v7wDataset('val', adaptive=True, max_boxes=max_boxes, maxlen=max_length)
    vqa2_dset = VQADataset(maxlen=max_length)
    tdiuc_dset = TDIUCDataset('train')
    pretrain_dset = ConcatDataset([v7w_train_dset, v7w_val_dset, vqa2_dset, tdiuc_dset])    
    print('train dataset size:', len(pretrain_dset))
    trainloader = DataLoader(pretrain_dset, config.batch_size, shuffle=True, num_workers=8, collate_fn=trim_collate)

    for epoch in trange(config.epoch, desc='Epoch', ncols=100): 
        log = ''
        if epoch == 5:
            optim.param_groups[0]['lr'] *= 0.2
        model.train()
        total_loss = 0.
        for (question, ans, features, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids) in tqdm(trainloader, desc='Train epoch %d'%epoch, ncols=100):
            optim.zero_grad()
            question, ans, features = question.cuda(), ans.cuda(), features.cuda()
            q_mask, q_lm_label_ids = q_mask.cuda(), q_lm_label_ids.cuda()
            a_mask, a_lm_label_ids = a_mask.cuda(), a_lm_label_ids.cuda()
            loss = model(question, ans, features, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()
            total_loss += loss.item()
        log += 'epoch %d:\n' % epoch
        log += 'train loss: %f\n' % total_loss
        # model.eval()
        # total_loss = 0.
        # with torch.no_grad():
        #     for (question, ans, features, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids) in tqdm(valloader, ncols=100):
        #         question, ans, features = question.cuda(), ans.cuda(), features.cuda()
        #         q_mask, q_lm_label_ids = q_mask.cuda(), q_lm_label_ids.cuda()
        #         a_mask, a_lm_label_ids = a_mask.cuda(), a_lm_label_ids.cuda()
        #         loss = model(question, ans, features, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids)
        #         total_loss += loss.item()
        # log += 'valid loss: %f\n\n' % total_loss
        
        with open(os.path.join(config.output, 'log.txt'), 'a') as f:
            f.write(log)
        torch.save(model.state_dict(), os.path.join(config.output, 'epoch%d.pth'%epoch))
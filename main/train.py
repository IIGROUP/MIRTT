import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from transformers  import BertTokenizer
from torchvision import transforms
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
import numpy as np
from datetime import datetime

from predict import get_predictions, compute_score
from model import TriTransModel

import config as argumentparser

if __name__ == '__main__':

    config = argumentparser.ArgumentParser()

    print("PyTorch version: ", torch.__version__)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Setting
    max_length = [config.qst_len, config.ans_len]

    if config.dataset == 'v7w':
        from dataset.v7w_dataset import v7wDataset, trim_collate
        print('Used the v7w dataset')
        max_boxes = 50
        sampler = None
        train_dset = v7wDataset('train', adaptive=True, max_boxes=max_boxes, maxlen=max_length)
        val_dset = v7wDataset('val', adaptive=True, max_boxes=max_boxes, maxlen=max_length)
        test_dset = v7wDataset('test', adaptive=True, max_boxes=max_boxes, maxlen=max_length)
        trainval_dset = ConcatDataset([train_dset, val_dset])

        trainloader = DataLoader(trainval_dset, config.batch_size, shuffle=sampler, num_workers=8,
                                    collate_fn=trim_collate, pin_memory=True)
        testloader = DataLoader(test_dset, config.batch_size, shuffle=False, sampler=sampler, num_workers=8,
                                    collate_fn=trim_collate, pin_memory=False)
    elif config.dataset == 'vqa1':
        from dataset.vqa1_dataset import trainDataset
        print('Used the vqa1.0 dataset')
        train_dset = trainDataset(max_length)
        trainloader = DataLoader(train_dset, config.batch_size, num_workers=8, drop_last=False, pin_memory=False)
    elif config.dataset == 'vqa2':
        from dataset.vqa2_dataset import vqaDataset
        print('Used the vqa2.0 dataset')
        trainset = vqaDataset('train,nominival', max_length)
        trainloader = DataLoader(
            trainset, batch_size=config.batch_size,
            shuffle=True, num_workers=8,
            drop_last=False, pin_memory=True
        )
        valset = vqaDataset('minival', max_length)
        valloader = DataLoader(
            valset, batch_size=config.batch_size,
            shuffle=True, num_workers=8,
            drop_last=False, pin_memory=True
        )
    elif config.dataset == 'tdiuc':
        from dataset.tdiuc_dataset import tdiuc_Dataset
        print('Used the tdiuc dataset')
        train_dset = tdiuc_Dataset('train', maxlen=max_length)
        val_dset = tdiuc_Dataset('val', maxlen=max_length)
        trainloader = DataLoader(train_dset, config.batch_size, num_workers=8, drop_last=False,
                                    collate_fn=trim_collate, pin_memory=False)
        valloader = DataLoader(val_dset, config.batch_size, shuffle=False, num_workers=8, drop_last=False,
                                    collate_fn=trim_collate, pin_memory=False)
    elif config.dataset == 'gqa':
        from dataset.gqa_dataset import gqaDataset 
        print('Used the gqa dataset')
        train_dset = gqaDataset('train', max_length)
        val_dset = gqaDataset('testdev', max_length)
        trainloader = DataLoader(train_dset, config.batch_size, num_workers=8, drop_last=False, pin_memory=False)
        valloader = DataLoader(val_dset, config.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)
    else:
        print('PLZ, select the right dataset!')
        os._exit(0)

    print("Successfully loaded data.")

    model = TriTransModel(num_layers=config.layers, ff_dim=config.ff_dim, dropout=config.dropout)   
    if config.load:
        print('Loading pretrain ckpt {}'.format(config.load))
        ckpt = torch.load(config.load)
        mod_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in mod_dict:
                mod_dict[k] = v
        model.load_state_dict(mod_dict)
        print('Finish!')
    if config.cuda and torch.cuda.is_available():
        model.cuda()

    # optimizer
    # gamma = 0.1 for lr schduler
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    lr_decay_step = 2
    lr_decay_rate = 0.25
    gradual_warmup_steps = [0.5 * config.lr, 1.0 * config.lr, 1.5 * config.lr, 2.0 * config.lr]
    lr_decay_epochs = range(10, 20, lr_decay_step)

    # Train the model
    best_score = 0
    for epoch in trange(config.epoch, desc='Epoch', ncols=100):
        model.train()
        run_bce_loss = 0
        total_score = 0

        log = '[epoch %d] ===========================================\n' % (epoch)

        if epoch < len(gradual_warmup_steps):
            optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            log += 'gradual warmup lr: %.8f\n' % optimizer.param_groups[0]['lr']
        elif epoch in lr_decay_epochs:
            optimizer.param_groups[0]['lr'] *= lr_decay_rate
            log += 'decreased lr: %.8f\n' % optimizer.param_groups[0]['lr']
        else:
            log += 'lr: %.8f\n' % optimizer.param_groups[0]['lr']

        process_bar = tqdm(trainloader, desc='Train', ncols=100)
        
        for (_, qst_ids, ans_ids, img_tensor, labels) in process_bar:
            
            qst_ids, ans_ids, img_tensor, labels = qst_ids.cuda(), ans_ids.cuda(), img_tensor.cuda(), labels.cuda()
            
            if config.dataset == 'v7w':
                qst_ids = qst_ids.unsqueeze(1).expand(qst_ids.size(0), 4, qst_ids.size(1)).contiguous().view(qst_ids.size(0) * 4, qst_ids.size(1))
                img_tensor = img_tensor.unsqueeze(1).expand(img_tensor.size(0), 4, img_tensor.size(1), img_tensor.size(2)).contiguous().view(img_tensor.size(0) * 4, img_tensor.size(1), img_tensor.size(2))
                ans_ids = ans_ids.view(ans_ids.size(0) * ans_ids.size(1), ans_ids.size(2))
                labels = labels.view(ans_ids.size(0), 1)
                labels = torch.cat([labels, 1-labels], 1)
                labels = labels.float()
            elif config.dataset == 'vqa1':
                qst_ids = qst_ids.unsqueeze(1).expand(qst_ids.size(0), 18, qst_ids.size(1)).contiguous().view(qst_ids.size(0) * 18, qst_ids.size(1))
                img_tensor = img_tensor.unsqueeze(1).expand(img_tensor.size(0), 18, img_tensor.size(1), img_tensor.size(2)).contiguous().view(img_tensor.size(0) * 18, img_tensor.size(1), img_tensor.size(2))
                ans_ids = ans_ids.view(ans_ids.size(0) * 18, ans_ids.size(2))
                labels = labels.view(labels.size(0) * 18, labels.size(2))       
            elif config.dataset == 'vqa2' or config.dataset == 'tdiuc' or config.dataset == 'gqa':
                qst_ids = qst_ids.unsqueeze(1).expand(qst_ids.size(0), 4, qst_ids.size(1)).contiguous().view(qst_ids.size(0) * 4, qst_ids.size(1))
                img_tensor = img_tensor.unsqueeze(1).expand(img_tensor.size(0), 4, img_tensor.size(1), img_tensor.size(2)).contiguous().view(img_tensor.size(0) * 4, img_tensor.size(1), img_tensor.size(2))
                ans_ids = ans_ids.view(ans_ids.size(0) * 4, ans_ids.size(2))
                labels = labels.view(labels.size(0) * 4, labels.size(2))    

            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=qst_ids, 
                            ans_ids=ans_ids,
                            labels=labels,
                            img_data=img_tensor)

            bce_loss = outputs[0]
            out_pred = outputs[1]

            bce_loss.backward()
            optimizer.step()

            # Train acc
            if config.dataset == 'vqa1':
                scores, batch_score, prob_preds = compute_score(out_pred, labels, 18)
            else:
                scores, batch_score, prob_preds = compute_score(out_pred, labels, 4)
            total_score += scores
            run_bce_loss += bce_loss.item()
            
            process_bar.set_postfix(BCE_loss=bce_loss.item(), batch_acc=batch_score)
            process_bar.update()

            
        train_bce_loss = run_bce_loss / len(trainloader)
        train_acc =  total_score / len(trainloader.dataset)
        log += 'Train acc: %.4f, Loss: %.4f\n' % (train_acc, train_bce_loss)

        if config.save_model:
            torch.save(model.state_dict(), os.path.join(config.output, 'epoch%d.pth'%epoch))

        # valid
        if config.dataset == 'v7w':
            print('Evaluating...')
            # Evaluate
            test_acc, test_bce_loss = get_predictions(config, model, testloader)

            log += 'Test acc: %.4f, Loss: %.4f\n' % (test_acc, test_bce_loss)

            # record the best
            if best_score < test_acc:
                best_score = test_acc
            log += 'Best score: %.4f\n' % best_score

        elif config.dataset == 'tdiuc':
            val_anslist = json.load(open('/home/joel-zhang/file/tdiuc/ban_ans/val_anslist.json'))
            ans_dict = {datum['question_id']: list(datum['answer'].keys()) for datum in val_anslist}
            targets = cPickle.load(open('/home/joel-zhang/file/tdiuc/cache/val_target.pkl', 'rb'))
            labels = {datum['question_id']: datum['labels'] for datum in targets}
            print('Evaluating...')
            # Evaluate
            val_acc = 0
            num = 0
            model.eval()
            for (q_id, qst_ids, ans_ids, img_tensor, _) in tqdm(valloader):
                qst_ids, img_tensor, ans_ids = qst_ids.cuda(), img_tensor.cuda(), ans_ids.cuda()
                
                qst_ids = qst_ids.unsqueeze(1).expand(qst_ids.size(0), 4, qst_ids.size(1)).contiguous().view(qst_ids.size(0) * 4, qst_ids.size(1))
                img_tensor = img_tensor.unsqueeze(1).expand(img_tensor.size(0), 4, img_tensor.size(1), img_tensor.size(2)).contiguous().view(img_tensor.size(0) * 4, img_tensor.size(1), img_tensor.size(2))
                ans_ids = ans_ids.view(ans_ids.size(0) * 4, ans_ids.size(2))

                with torch.no_grad():
                    outputs = model(input_ids=qst_ids, 
                                    ans_ids=ans_ids,
                                    img_data=img_tensor)
                pred = outputs[0]
                pred = pred.reshape(-1, 4, 2)
                for i in range(pred.size(0)):
                    try:
                        s = pred[i][:, 0]
                        _, ans_id = s.max(0)
                        ans = ans_dict[q_id[i].item()][ans_id]            
                        target_ans = label2ans[labels[q_id[i].item()]]
                        if ans == target_ans:
                            val_acc += 1
                        num += 1
                    except:
                        print('val error')

            val_acc = val_acc / num
            log += "tritrans Valid acc: %f\n" % val_acc

        elif config.dataset == 'vqa1':
            pass
        
        elif config.dataset == 'vqa2':
            print('Evaluating...')
            # Evaluate
            model.eval()

            quesid2ans = {}
            ans_list = json.load(open('/home/jyt21/VQA/MIRTT/bilinear_method/result/bertban_vqa2/train_anslist.json'))
            ans_dict = {datum['question_id']: list(datum['answer'].keys()) for datum in ans_list}

            with torch.no_grad():
                for (q_id, ques, ans, img) in tqdm(valloader):
                    ques, img, ans = ques.cuda(), img.cuda(), ans.cuda()
                    ques = ques.unsqueeze(1).expand(ques.size(0), 4, ques.size(1)).contiguous().view(ques.size(0) * 4, ques.size(1))
                    img = img.unsqueeze(1).expand(img.size(0), 4, img.size(1), img.size(2)).contiguous().view(img.size(0) * 4, img.size(1), img.size(2))
                    ans = ans.view(ans.size(0) * 4, ans.size(2))

                    outputs = model(input_ids=ques, ans_ids=ans, img_data=img)
                    pred = outputs[0]
                    pred = pred.reshape(-1, 4, 2)

                    for i in range(pred.size(0)):
                        try:
                            s = pred[i][:, 0]
                            _, ans_id = s.max(0)
                            ans = ans_dict[q_id[i].item()][ans_id]
                            quesid2ans[q_id[i].item()] = ans
                        except:
                            print('val error')
            
            queslist = json.load(open('/mnt/data/jyt/VQA2.0/minival.json', 'r'))
            id2datum = {
                datum['question_id']: datum
                for datum in queslist
            }
            score = 0.
            for quesid, ans in quesid2ans.items():
                datum = id2datum[quesid]
                label = datum['label']
                if ans in label:
                    score += label[ans]
            log += "tritrans valid acc: %f\n" % (score / len(quesid2ans))

        elif config.dataset == 'gqa':
            print('Evaluating...')
            # Evaluate
            model.eval()

            quesid2ans = {}
            ans_list = json.load(open('/home/joel-zhang/file/jyt/bi_gqa/result/ban/test_anslist.json'))
            ans_dict = {datum['question_id']: list(datum['answer'].keys()) for datum in ans_list}

            with torch.no_grad():
                for (q_id, ques, ans, img) in tqdm(valloader):
                    ques, img, ans = ques.cuda(), img.cuda(), ans.cuda()
                    ques = ques.unsqueeze(1).expand(ques.size(0), 4, ques.size(1)).contiguous().view(ques.size(0) * 4, ques.size(1))
                    img = img.unsqueeze(1).expand(img.size(0), 4, img.size(1), img.size(2)).contiguous().view(img.size(0) * 4, img.size(1), img.size(2))
                    ans = ans.view(ans.size(0) * 4, ans.size(2))

                    outputs = model(input_ids=ques, ans_ids=ans, img_data=img)
                    pred = outputs[0]
                    pred = pred.reshape(-1, 4, 2)

                    for i in range(pred.size(0)):
                        try:
                            s = pred[i][:, 0]
                            _, ans_id = s.max(0)
                            ans = ans_dict[q_id[i]][ans_id]     
                            quesid2ans[q_id[i]] = ans
                        except:
                            print('val error')
            
            queslist = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/testdev.json', 'r'))
            id2datum = {
                datum['question_id']: datum
                for datum in queslist
            }
            score = 0.
            for quesid, ans in quesid2ans.items():
                datum = id2datum[quesid]
                label = datum['label']
                if ans in label:
                    score += label[ans]
            log += "tritrans valid acc: %f\n" % (score / len(quesid2ans))

        with open(os.path.join(config.output, 'log.txt'), 'a') as f:
            f.write(log)
        



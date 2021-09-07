import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import os
import datetime, time
import numpy as np
import warnings
import _pickle as cPickle
import json
warnings.filterwarnings("ignore")

from bi_model import Net
from configs import C


if __name__ == '__main__':

    if C.mod == 'train':

        # dataset
        if C.dataset == 'vqa2':
            label2ans = json.load(open('/mnt/data/jyt/VQA2.0/trainval_label2ans.json'))
            ans_size = len(label2ans)

            from dataset.vqa2_dataset import vqaDataset
            trainset = vqaDataset('train,nominival', C.MAX_TOKEN)
            train_loader = DataLoader(
                trainset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )
            valset = vqaDataset('minival', C.MAX_TOKEN)
            val_loader = DataLoader(
                valset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )

        elif C.dataset == 'tdiuc':
            label2ans = cPickle.load(open('/home/joel-zhang/file/tdiuc/cache/trainval_label2ans.pkl', 'rb'))
            ans_size = len(label2ans)

            from dataset.tdiuc_dataset import tdiuc_Dataset, trim_collate
            trainset = tdiuc_Dataset('train', C.MAX_TOKEN)
            train_loader = DataLoader(
                trainset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, collate_fn=trim_collate, pin_memory=True
            )
            valset = tdiuc_Dataset('val', C.MAX_TOKEN)
            val_loader = DataLoader(
                valset, batch_size=C.batch_size,
                shuffle=False, num_workers=C.num_workers,
                drop_last=False, collate_fn=trim_collate, pin_memory=True
            )

        elif C.dataset == 'gqa':
            label2ans = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/trainval_label2ans.json'))
            ans_size = len(label2ans)

            from dataset.gqa_dataset import trainDataset, testDataset
            trainset = trainDataset(C.MAX_TOKEN)
            train_loader = DataLoader(
                trainset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )
            valset = testDataset(C.MAX_TOKEN)
            val_loader = DataLoader(
                valset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )

        # model
        model = Net(C, ans_size)
        if C.load:
            print('Loading ckpt {}'.format(C.load))
            ckpt = torch.load(C.load)            
            model.load_state_dict(ckpt)
            print('Finish!')
        model = model.cuda()
        if C.multiGPU:
            model = nn.DataParallel(model, C.gpuID)

        # optimizer
        bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=C.LR_BASE) 
        gradual_warmup_steps = [0.5 * C.LR_BASE, 1.0 * C.LR_BASE, 1.5 * C.LR_BASE, 2.0 * C.LR_BASE]

        for epoch in range(C.epochs):
            if epoch < len(gradual_warmup_steps):
                optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            elif epoch in C.LR_DECAY_LIST:
                optim.param_groups[0]['lr'] *= C.LR_DECAY_R
                
            log_str = "--------------------------------\n"
            log_str += "Epoch %d:\n" % epoch
            log_str += 'nowTime: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n'
            log_str += 'learning rate: %f\n' % optim.param_groups[0]['lr']
            time_start = time.time()
            loss_sum = 0

            model.train()
            for (ques_id, feats, ques, target) in tqdm(train_loader, desc='epoch%d '%epoch):
                
                optim.zero_grad()

                feats, ques, target = feats.cuda(), ques.cuda(), target.cuda()
                pred = model(feats, ques)

                loss = bce_loss(pred, target)
                loss_sum += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optim.step()

            time_end = time.time()
            log_str += 'Train cost {} min\n'.format(int((time_end-time_start)/60))
            log_str += "Train loss %f\n" % loss_sum
            print('\nlearning rate: %f\n' % optim.param_groups[0]['lr'])
            print('Train cost {} min\n'.format(int((time_end-time_start)/60)))
            print("Train loss %f\n" % loss_sum)

            # valid
            model.eval()
            if C.dataset == 'tdiuc':
                from dataset.tdiuc_dataset import evaluate
                valid_score, upper_bound = evaluate(model, val_loader)
                log_str += "Valid score %f\n" % valid_score
                log_str += "upper bound %f\n" % upper_bound
                print("Valid acc: %f\n" % (valid_score * 100.))

            else:
                quesid2ans = {}
                for (ques_id, feats, ques, target) in tqdm(val_loader):
                    with torch.no_grad():
                        feats, ques = feats.cuda(), ques.cuda()
                        logit = model(feats, ques)
                        score, label = logit.max(1)
                        for qid, l in zip(ques_id, label.cpu().numpy()):
                            ans = label2ans[l]
                            quesid2ans[qid.item()] = ans
                score = 0.
                if C.dataset == 'gqa':
                    queslist = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/testdev.json', 'r'))
                elif C.dataset == 'vqa2':
                    queslist = json.load(open('/mnt/data/jyt/VQA2.0/minival.json', 'r'))
                id2datum = {
                    datum['question_id']: datum
                    for datum in queslist
                }
                for quesid, ans in quesid2ans.items():
                    datum = id2datum[quesid]
                    label = datum['label']
                    if ans in label:
                        score += label[ans]
                log_str += "Valid score %f\n" % (score / len(quesid2ans))
                print('valid score:', score / len(quesid2ans))

            with open(C.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

            torch.save(model.state_dict(), os.path.join(C.output, "epoch%d.pth"%epoch))
            

    elif C.mod == 'test':  # for vqa2.0 and gqa online evaluation

        if C.dataset == 'vqa2':
            label2ans = json.load(open('/mnt/data/jyt/VQA2.0/trainval_label2ans.json'))
            ans_size = len(label2ans)

            from dataset.vqa2_dataset import vqaDataset
            testset = vqaDataset('test', C.MAX_TOKEN)
            test_loader = DataLoader(
                testset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )

        elif C.dataset == 'gqa':
            label2ans = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/trainval_label2ans.json'))
            ans_size = len(label2ans)

            from dataset.gqa_dataset import submitDataset
            testset = submitDataset(C.MAX_TOKEN)
            test_loader = DataLoader(
                testset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )

        # model
        model = Net(C, ans_size)
        if C.load:
            print('Loading ckpt {}'.format(C.load))
            ckpt = torch.load(C.load)            
            model.load_state_dict(ckpt)
            print('Finish!')
        model = model.cuda()
        if C.multiGPU:
            model = nn.DataParallel(model, C.gpuID)

        model.eval()
        id2ans = []
        for (ques_id, feats, ques) in tqdm(test_loader):
            with torch.no_grad():
                feats, ques = feats.cuda(), ques.cuda()
                logit = model(feats, ques)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = label2ans[l]
                    if C.dataset == 'vqa2':
                        id2ans.append({
                            'question_id': qid.item(),
                            'answer': ans
                        })
                    elif C.dataset == 'gqa':
                        id2ans.append({
                            'questionId': qid.item(),
                            'prediction': ans
                        })
        with open(C.output + '/submit.json', 'w') as f:
            json.dump(id2ans, f, indent=4, sort_keys=True)

    elif C.mod == 'generate':  # generate answer list
        if C.dataset == 'vqa2':
            label2ans = json.load(open('/mnt/data/jyt/VQA2.0/trainval_label2ans.json'))
            ans_size = len(label2ans)

            from dataset.vqa2_dataset import vqaDataset
            trainset = vqaDataset('train,nominival,minival', C.MAX_TOKEN)
            train_loader = DataLoader(
                trainset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )
            testset = vqaDataset('test', C.MAX_TOKEN)
            test_loader = DataLoader(
                testset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )

        elif C.dataset == 'tdiuc':
            label2ans = cPickle.load(open('/home/joel-zhang/file/tdiuc/cache/trainval_label2ans.pkl', 'rb'))
            ans_size = len(ans2label)
            from dataset.tdiuc_dataset import tdiuc_Dataset, trim_collate
            trainset = tdiuc_Dataset('train', C.MAX_TOKEN)
            train_loader = DataLoader(
                trainset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, collate_fn=trim_collate, pin_memory=True
            )
            valset = tdiuc_Dataset('val', C.MAX_TOKEN)
            val_loader = DataLoader(
                valset, batch_size=C.batch_size,
                shuffle=False, num_workers=C.num_workers,
                drop_last=False, collate_fn=trim_collate, pin_memory=True
            )

        elif C.dataset == 'gqa':
            label2ans = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/trainval_label2ans.json'))
            ans_size = len(label2ans)

            from dataset.gqa_dataset import trainDataset, testDataset, submitDataset
            trainset = trainDataset(C.MAX_TOKEN)
            train_loader = DataLoader(
                trainset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )
            valset = testDataset(C.MAX_TOKEN)
            val_loader = DataLoader(
                valset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )
            testset = submitDataset(C.MAX_TOKEN)
            test_loader = DataLoader(
                testset, batch_size=C.batch_size,
                shuffle=True, num_workers=C.num_workers,
                drop_last=False, pin_memory=True
            )

        model = Net(C, ans_size)
        if C.load:
            print('Loading ckpt {}'.format(C.load))
            ckpt = torch.load(C.load)            
            model.load_state_dict(ckpt)
            print('Finish!')
        model = model.cuda()
        model.eval()

        if C.dataset == 'vqa2':
            id2ans = []
            for (ques_id, feats, ques, target) in tqdm(train_loader):
                feats, ques, target = feats.cuda(), ques.cuda(), target.cuda()
                pred = model(feats, ques)
                for i in range(pred.size(0)):
                    top4 = torch.topk(pred[i], 4).indices
                    top4 = top4.cpu().numpy().tolist()
                    d = {}
                    for j in top4:
                        a = label2ans[j]
                        s = target[i][j].item()
                        if 0 <= s and s <= 0.1:
                            d[a] = [0,1]
                        elif 0.95 <= s and s <= 1:
                            d[a] = [1,0]
                        elif 0.25 <= s and s <= 0.35:
                            d[a] = [0.7,0.3]
                        elif 0.55 <= s and s <= 0.65:
                            d[a] = [0.8,0.2]
                        elif 0.85 <= s and s <= 0.95:
                            d[a] = [0.9,0.1]
                        else:
                            d[a] = [0.5, 0.5]
                    id2ans.append({
                        'question_id': ques_id[i].item(),
                        'answer': d
                    })
            with open(os.path.join(C.output, 'train_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)
            id2ans = []
            for (ques_id, feats, ques) in tqdm(test_loader):
                feats, ques = feats.cuda(), ques.cuda()
                pred = model(feats, ques)
                for i in range(pred.size(0)):
                    score = F.softmax(pred[i])
                    top4 = torch.topk(score, 4).indices
                    top4 = top4.cpu().numpy().tolist()
                    d = {}
                    for j in top4:
                        a = label2ans[j]
                        s = score[j].item()
                        d[a] = s
                    id2ans.append({
                        'question_id': ques_id[i].item(),
                        'answer': d
                    })
            with open(os.path.join(C.output, 'test_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)

        elif C.dataset == 'tdiuc':
            id2ans = []
            for (q_id, v, q, a) in tqdm(train_loader):
                v = v.cuda()
                q = q.cuda()
                with torch.no_grad():
                    preds = model(v, q)
                    for i in range(preds.size(0)):
                        top4 = torch.topk(preds[i], 4).indices
                        top4 = top4.cpu().numpy().tolist()
                        d = {}
                        for j in top4:
                            ans = label2ans[j]
                            s = a[i][j].item()
                            if s == 0:
                                d[ans] = [0,1]
                            elif s == 1:
                                d[ans] = [1,0]
                            else:
                                print(s)
                        id2ans.append({
                            'question_id': q_id[i].item(),
                            'answer': d
                        })
            with open(os.path.join(C.output, 'train_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)
            id2ans = []
            for (ques_id, feats, ques) in tqdm(test_loader):
                feats, ques = feats.cuda(), ques.cuda()
                pred = model(feats, ques)
                for i in range(pred.size(0)):
                    score = F.softmax(pred[i])
                    top4 = torch.topk(score, 4).indices
                    top4 = top4.cpu().numpy().tolist()
                    d = {}
                    for j in top4:
                        a = label2ans[j]
                        s = score[j].item()
                        d[a] = s
                    id2ans.append({
                        'question_id': ques_id[i].item(),
                        'answer': d
                    })
            with open(os.path.join(C.output, 'test_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)

        elif C.dataset == 'gqa':
            id2ans = []
            for (q_id, v, q, a) in tqdm(train_loader):
                v = v.cuda()
                q = q.cuda()
                with torch.no_grad():
                    preds = model(v, q)
                    for i in range(preds.size(0)):
                        top4 = torch.topk(preds[i], 4).indices
                        top4 = top4.cpu().numpy().tolist()
                        d = {}
                        for j in top4:
                            ans = label2ans[j]
                            s = a[i][j].item()
                            if s == 0:
                                d[ans] = [0,1]
                            elif s == 1:
                                d[ans] = [1,0]
                            else:
                                print(s)
                        id2ans.append({
                            'question_id': q_id[i],
                            'answer': d
                        })
            with open(os.path.join(C.output, 'train_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)
            id2ans = []
            for (q_id, v, q, _) in tqdm(val_loader):
                v = v.cuda()
                q = q.cuda()
                with torch.no_grad():
                    preds = model(v, q)
                    for i in range(preds.size(0)):
                        top4 = torch.topk(preds[i], 4).indices
                        top4 = top4.cpu().numpy().tolist()
                        d = {}
                        for j in top4:
                            ans = label2ans[j]
                            s = preds[i][j].item()
                            d[ans] = s
                        id2ans.append({
                            'question_id': q_id[i],
                            'answer': d
                        })
            with open(os.path.join(C.output, 'test_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)
            id2ans = []
            for (q_id, v, q) in tqdm(test_loader):
                with torch.no_grad():
                    v, q = v.cuda(), q.cuda()
                    preds = model(v, q)
                    for i in range(preds.size(0)):
                        top4 = torch.topk(preds[i], 4).indices
                        top4 = top4.cpu().numpy().tolist()
                        d = {}
                        for j in top4:
                            ans = label2ans[j]
                            s = preds[i][j].item()
                            d[ans] = s
                        id2ans.append({
                            'question_id': q_id[i],
                            'answer': d
                        })
            with open(os.path.join(C.output, 'submit_anslist.json'), 'w') as f:
                json.dump(id2ans, f, indent=4, sort_keys=True)
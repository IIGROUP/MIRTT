import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

import os
import json
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score

import config as argumentparser
from model import TriTransModel


def compute_score(logits, labels, num=4):
    prob_preds = torch.softmax(logits, 1)
    prob_preds = prob_preds[:, 0].view(-1, num)
    labels = labels[:, 0].view(-1, num)
    batch_score = accuracy_score(torch.argmax(labels, dim=1).cpu(), torch.argmax(prob_preds, dim=1).cpu())
    scores = accuracy_score(torch.argmax(labels, dim=1).cpu(), torch.argmax(prob_preds, dim=1).cpu(), normalize=False)
    return scores, batch_score, prob_preds

def get_predictions(config, model, dataloader):
    
    # predictions = []
    total_score = 0
    model.eval()
    process_bar = tqdm(dataloader, desc='Evaluate', ncols=100) 
    run_bce_loss = 0
    index = 0
      
    with torch.no_grad():

        for (_, qst_ids, ans_ids, img_tensor, labels) in process_bar:

            bce_criterion = BCEWithLogitsLoss(reduction = 'mean')
                        
            qst_ids, ans_ids, img_tensor, labels = qst_ids.cuda(), ans_ids.cuda(), img_tensor.cuda(), labels.cuda()
            qst_ids = qst_ids.unsqueeze(1).expand(qst_ids.size(0), 4, qst_ids.size(1)).contiguous().view(qst_ids.size(0) * 4, qst_ids.size(1))
            ans_ids = ans_ids.view(ans_ids.size(0) * ans_ids.size(1), ans_ids.size(2))

            labels = labels.view(ans_ids.size(0), 1)
            labels = torch.cat([labels, 1-labels], 1)
            labels = labels.float()
            img_tensor = img_tensor.unsqueeze(1).expand(img_tensor.size(0), 4, img_tensor.size(1), img_tensor.size(2)).contiguous().view(img_tensor.size(0) * 4, img_tensor.size(1), img_tensor.size(2))

            outputs = model(input_ids=qst_ids, 
                        ans_ids=ans_ids,
                        labels=None,
                        img_data=img_tensor)
            
            logits = outputs[0]
            scores, batch_score, prob_preds = compute_score(logits, labels)
            total_score += scores

            bce_loss = bce_criterion(logits, labels)
            process_bar.set_postfix(BCE_loss=bce_loss.item(), batch_acc=batch_score)
            process_bar.update()

            run_bce_loss += bce_loss.item()
            index += 1

    acc = total_score / len(dataloader.dataset)
    test_bce_loss = run_bce_loss / len(dataloader)

    return acc, test_bce_loss

if __name__ == '__main__':
    config = argumentparser.ArgumentParser()

    max_length = [config.qst_len, config.ans_len]

    if config.dataset == 'vqa1':
        print('test on the vqa1.0 dataset')
        from dataset.vqa1_dataset import testDataset
        test_dset = testDataset(max_length)
        testloader = DataLoader(test_dset, config.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

        queslist = json.load(open('/home/joel-zhang/file/vqa1.0/QA/mc_test_ques.json', 'r'))['questions']
        ans_dict = {}
        for datum in queslist:
            ans_dict[datum['question_id']] = datum['multiple_choices']
    
    elif config.dataset == 'vqa2':
        print('test on the vqa2.0 dataset')
        from dataset.vqa2_dataset import vqaDataset
        test_dset = vqaDataset('test', max_length)
        testloader = DataLoader(test_dset, config.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

        ans_list = json.load(open('/home/jyt21/VQA/MIRTT/bilinear_method/result/bertban_vqa2/test_anslist.json'))
        ans_dict = {datum['question_id']: list(datum['answer'].keys()) for datum in ans_list}
        scr_dict = {datum['question_id']: list(datum['answer'].values()) for datum in ans_list} # baseline score(used in ensemble)

    elif config.dataset == 'gqa':
        print('test on the GQA dataset')
        from dataset.gqa_dataset import gqaDataset
        test_dset = vqaDataset('test', max_length)
        testloader = DataLoader(test_dset, config.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

        ans_list = json.load(open('/home/joel-zhang/file/jyt/vilbert_beta/vilbert_test_anslist.json'))
        ans_dict = {datum['question_id']: list(datum['answer'].keys()) for datum in ans_list}
        scr_dict = {datum['question_id']: list(datum['answer'].values()) for datum in ans_list}

    model = TriTransModel(num_layers=config.layers, ff_dim=config.ff_dim, dropout=config.dropout)

    if config.load:
        print('Loading pretrain ckpt {}'.format(config.load))
        ckpt = torch.load(config.load)
        model.load_state_dict(ckpt)
        print('Finish!')
    model = model.cuda()

    model.eval()

    quesid2ans = {}  

    with torch.no_grad():
        for (q_id, ques, ans, img) in tqdm(testloader):
            ques, img, ans = ques.cuda(), img.cuda(), ans.cuda()
            if config.dataset == 'vqa1':
                ques = ques.unsqueeze(1).expand(ques.size(0), 18, ques.size(1)).contiguous().view(ques.size(0) * 18, ques.size(1))
                img = img.unsqueeze(1).expand(img.size(0), 18, img.size(1), img.size(2)).contiguous().view(img.size(0) * 18, img.size(1), img.size(2))
                ans = ans.view(ans.size(0) * 18, ans.size(2))

                outputs = model(input_ids=ques, ans_ids=ans, img_data=img)
                pred = outputs[0]
                pred = pred.reshape(-1, 18, 2)

            elif config.dataset == 'vqa2' or config.dataset == 'gqa':
                ques = ques.unsqueeze(1).expand(ques.size(0), 4, ques.size(1)).contiguous().view(ques.size(0) * 4, ques.size(1))
                img = img.unsqueeze(1).expand(img.size(0), 4, img.size(1), img.size(2)).contiguous().view(img.size(0) * 4, img.size(1), img.size(2))
                ans = ans.view(ans.size(0) * 4, ans.size(2))

                outputs = model(input_ids=ques, ans_ids=ans, img_data=img)
                pred = outputs[0]
                pred = pred.reshape(-1, 4, 2)

            for i in range(pred.size(0)):
                score = pred[i][:, 0]
                # score -= score.min(0)[0]
                # score /= score.max(0)[0]
                # bs_score = scr_dict[q_id[i].item()]
                # bs_score = np.array(bs_score)
                # bs_score = torch.from_numpy(bs_score)
                # bs_score -= bs_score.min(0)[0]
                # bs_score /= bs_score.max(0)[0]
                # bs_score = bs_score.cuda()
                # score += bs_score
                _, ans_id = score.max(0)
                ans = ans_dict[q_id[i].item()][ans_id]
                quesid2ans[q_id[i].item()] = ans

    # with open(os.path.join(config.output, 'test_ensemble_predict.json'), 'w') as f:
    with open(os.path.join(config.output, 'test_predict.json'), 'w') as f:
        result = []
        for ques_id, ans in quesid2ans.items():
            result.append({
                'question_id': ques_id,
                'answer': ans
            })
        json.dump(result, f, indent=4, sort_keys=True)
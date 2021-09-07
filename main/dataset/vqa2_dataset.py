import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import h5py
import base64
import csv
import sys
import time
from transformers import BertTokenizer

QUESPATH = {
    'train': '/mnt/data/jyt/VQA2.0/train.json',
    # 'vg': '/mnt/data/jyt/VQA2.0/vg.json',
    'nominival': '/mnt/data/jyt/VQA2.0/nominival.json',
    'minival': '/mnt/data/jyt/VQA2.0/minival.json',
    'test': '/mnt/data/jyt/VQA2.0/test.json',
}
train_ans = json.load(open('/home/jyt21/VQA/MIRTT/bilinear_method/result/bertban_vqa2/train_anslist.json'))
val_ans = json.load(open('/home/jyt21/VQA/MIRTT/bilinear_method/result/bertban_vqa2/train_anslist.json'))
test_ans = json.load(open('/home/jyt21/VQA/MIRTT/bilinear_method/result/bertban_vqa2/test_anslist.json'))

train_quesid2ans = {datum['question_id']: datum['answer'] for datum in train_ans}
val_quesid2ans = {datum['question_id']: datum['answer'] for datum in val_ans}
test_quesid2ans = {datum['question_id']: datum['answer'] for datum in test_ans}
# for datum in ans_list:
#     flag = 1
#     for k in datum['answer'].keys():
#         if len(k) == 0:
#             flag = 0
#             break
#     if flag:
#         quesid2ans[datum['question_id']] = datum['answer']


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = {}
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):            
            boxes = int(item['num_boxes'])
            feat = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32)
            feat = feat.reshape((boxes, -1))
            id = item['img_id']
            data[id] = feat
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


class vqaDataset(Dataset):
    def __init__(self, splits, maxlen):
        
        # questions
        queslist = []
        for split in splits.split(','):
            queslist.extend(json.load(open(QUESPATH[split])))            
        # images
        self.imgdata = {}
        if splits == 'test':
            self.imgdata.update(load_obj_tsv('/mnt/data/jyt/VQA2.0/mscoco_imgfeat/test2015_obj36.tsv'))
        elif splits == 'minival':
            self.imgdata.update(load_obj_tsv('/mnt/data/jyt/VQA2.0/mscoco_imgfeat/val2014_obj36.tsv'))
        else:
            self.imgdata.update(load_obj_tsv('/mnt/data/jyt/VQA2.0/mscoco_imgfeat/train2014_obj36.tsv'))
            self.imgdata.update(load_obj_tsv('/mnt/data/jyt/VQA2.0/mscoco_imgfeat/val2014_obj36.tsv'))

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Only keep the data with loaded image features
        self.quesdata = []  # question information
        for datum in queslist:
            if datum['img_id'] in self.imgdata: # or 'COCO_train2014_000000'+str(datum['img_id']) in self.imgdata or 'COCO_val2014_000000'+str(datum['img_id']) in self.imgdata:
                self.quesdata.append(datum)
        print("%s: Use %d data in dataset\n" % (splits, len(self.quesdata)))

        self.maxlen = maxlen
        self.splits = splits

    def __len__(self):
        return len(self.quesdata)

    def proc_query(self, text, max_seq_length):
        tokens = self.tokenizer.tokenize(text.strip())

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if len(tokens) < max_seq_length:
            tokens += ['[PAD]' for _ in range(max_seq_length - len(tokens))] #Padding sentences

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return torch.tensor(input_ids)

    def __getitem__(self, item: int):
        datum = self.quesdata[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        ques_ids = self.proc_query(ques, self.maxlen[0])

        # if isinstance(img_id, int): # from vg
        #     if 'COCO_train2014_000000'+str(img_id) in self.imgdata:
        #         feats = self.imgdata['COCO_train2014_000000'+str(img_id)]
        #     else:
        #         feats = self.imgdata['COCO_val2014_000000'+str(img_id)]
        # else:
        #     feats = self.imgdata[img_id]
        feats = self.imgdata[img_id]
        feats = torch.from_numpy(feats)

        if self.splits == 'minival':
            ansdict = val_quesid2ans[ques_id]
            ans = []
            for k in ansdict:
                a = self.proc_query(k, self.maxlen[1])
                ans.append(a)
            answer = torch.stack(ans, 0)
            return ques_id, ques_ids, answer, feats.float()
        elif self.splits == 'test':
            ansdict = test_quesid2ans[ques_id]
            ans = []
            for k, v in ansdict.items():
                a = self.proc_query(k, self.maxlen[1])
                ans.append(a)
            answer = torch.stack(ans, 0)
            return ques_id, ques_ids, answer, feats.float()
        else:
            ansdict = train_quesid2ans[ques_id]
            ans = []
            score = []
            for k, v in ansdict.items():
                a = self.proc_query(k, self.maxlen[1])
                ans.append(a)
                score.append(v)
            answer = torch.stack(ans, 0)
            score = np.array(score)
            score = torch.from_numpy(score)
        
            return ques_id, ques_ids, answer, feats.float(), score.float()

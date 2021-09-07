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


class trainDataset(Dataset):
    def __init__(self, maxlen):

        # quesitons
        queslist = \
            json.load(open('/home/joel-zhang/file/vqa1.0/QA/train.json', 'r')) + \
            json.load(open('/home/joel-zhang/file/vqa1.0/QA/val.json', 'r'))

        # images
        self.imgdata = {}
        self.imgdata.update(load_obj_tsv('/home/joel-zhang/file/lxmert/data/vqa/img_feat/train2014_obj36.tsv'))
        self.imgdata.update(load_obj_tsv('/home/joel-zhang/file/lxmert/data/vqa/img_feat/val2014_obj36.tsv'))
                
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Only keep the data with loaded image features
        self.quesdata = []  # question information
        for datum in queslist:
            if datum['image_id'] in self.imgdata:
                self.quesdata.append(datum)
        print("%s: Use %d data in dataset\n" % ('train', len(self.quesdata)))

        self.maxlen = maxlen

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

        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']
        ques = self.proc_query(ques, self.maxlen[0])
        
        feats = self.imgdata[img_id]

        ansdict = datum['answer']
        ans = []
        score = []
        for k, v in ansdict.items():
            a = self.proc_query(k, self.maxlen[1])
            ans.append(a)
            score.append(v)
        answer = torch.stack(ans, 0)
        score = np.array(score)
        score = torch.from_numpy(score)

        return ques_id, ques, answer, torch.from_numpy(feats).float(), score


class testDataset(Dataset):
    def __init__(self, maxlen):

        # quesitons
        queslist = \
            json.load(open('/home/joel-zhang/file/vqa1.0/QA/mc_test_ques.json', 'r'))['questions']
        for datum in queslist:
            img_id = str(datum['image_id'])
            while len(img_id) < 12:
                img_id = '0' + img_id
            img_id = 'COCO_test2015_' + img_id
            datum['image_id'] = img_id

        # images
        self.imgdata = {}
        self.imgdata.update(load_obj_tsv('/home/joel-zhang/file/lxmert/data/vqa/img_feat/test2015_obj36.tsv'))
                
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Only keep the data with loaded image features
        self.quesdata = []  # question information
        for datum in queslist:
            if datum['image_id'] in self.imgdata:
                self.quesdata.append(datum)
        print("%s: Use %d data in dataset\n" % ('test', len(self.quesdata)))

        self.maxlen = maxlen

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

        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']
        ques = self.proc_query(ques, self.maxlen[0])
        
        feats = self.imgdata[img_id]

        anslist = datum['multiple_choices']
        ans = []
        for k in anslist:
            a = self.proc_query(k, self.maxlen[1])
            ans.append(a)
        answer = torch.stack(ans, 0)

        return ques_id, ques, answer, torch.from_numpy(feats).float()

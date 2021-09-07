import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import json
import csv
import time
import base64
import sys
import _pickle as cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
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



class gqaDataset(Dataset):
    def __init__(self, name, maxlen):

        if name == 'train':
            # quesitons
            queslist = \
                json.load(open('/home/joel-zhang/file/lxmert/data/gqa/train.json', 'r')) + \
                json.load(open('/home/joel-zhang/file/lxmert/data/gqa/valid.json', 'r'))
            # images
            self.imgdata = {}
            train_feat_path = '/home/joel-zhang/file/lxmert/data/vg_gqa_imgfeat/vg_gqa_obj36.tsv'
            self.imgdata.update(load_obj_tsv(train_feat_path))
            anslist = json.load(open('/home/joel-zhang/file/jyt/bi_gqa/result/ban/train_anslist.json'))
        elif name == 'testdev':
            # quesitons
            queslist = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/testdev.json', 'r'))
            # images
            self.imgdata = {}
            train_feat_path = '/home/joel-zhang/file/lxmert/data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv'
            self.imgdata.update(load_obj_tsv(train_feat_path))
            anslist = json.load(open('/home/joel-zhang/file/jyt/bi_gqa/result/ban/test_anslist.json'))
        elif name == 'submit':
            # quesitons
            queslist = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/submit.json', 'r'))
            # images
            self.imgdata = {}
            train_feat_path = '/home/joel-zhang/file/lxmert/data/vg_gqa_imgfeat/vg_gqa_obj36.tsv'
            self.imgdata.update(load_obj_tsv(train_feat_path))
            anslist = json.load(open('/home/joel-zhang/file/jyt/bi_gqa/result/ban/submit_anslist.json'))

        # answer
        self.ans2label = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/trainval_ans2label.json'))
        self.label2ans = json.load(open('/home/joel-zhang/file/lxmert/data/gqa/trainval_label2ans.json'))
        assert len(self.ans2label) == len(self.label2ans)
        self.ans_size = len(self.ans2label)
        print('answer size: ', self.ans_size)
        
        self.quesid2ans = {datum['question_id']: datum['answer'] for datum in anslist}

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Only keep the data with loaded image features
        self.quesdata = []  # question information
        for datum in queslist:
            if datum['img_id'] in self.imgdata:
                self.quesdata.append(datum)
        print("%s: Use %d data in dataset\n" % ('train', len(self.quesdata)))

        self.maxlen = maxlen
        self.name = name

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
        ques = self.proc_query(ques, self.maxlen[0])
        
        feats = self.imgdata[img_id]

        ansdict = self.quesid2ans[ques_id]
        ans = []
        score = []
        for k, v in ansdict.items():
            a = self.proc_query(k, self.maxlen[1])
            ans.append(a)
            score.append(v)
        answer = torch.stack(ans, 0)
        score = np.array(score)
        score = torch.from_numpy(score)
        if self.name == 'train':
            return ques_id, ques, answer, torch.from_numpy(feats).float(), score
        else:
            return ques_id, ques, answer, torch.from_numpy(feats).float()
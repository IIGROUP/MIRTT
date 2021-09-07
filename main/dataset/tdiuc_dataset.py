import torch
from torch.utils.data import Dataset
from torch._six import string_classes
import torch.nn.functional as F

from __future__ import print_function
import os
import json
import h5py
import _pickle as cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import h5py
import itertools
from transformers import BertTokenizer
import collections


def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

QUESPATH = {
    'train': '/home/joel-zhang/file/tdiuc/TDIUC_train_questions.json',
    'val': '/home/joel-zhang/file/tdiuc/TDIUC_val_questions.json'
}
IMGPATH= {
    'train': '/home/joel-zhang/file/tdiuc/train.hdf5',
    'val': '/home/joel-zhang/file/tdiuc/val.hdf5'
}
IMGIDPATH= {
    'train': '/home/joel-zhang/file/tdiuc/train_imgid2idx.pkl',
    'val': '/home/joel-zhang/file/tdiuc/val_imgid2idx.pkl'
}
ANSPTH = {
    'train': '/home/joel-zhang/file/tdiuc/ban_ans/train_anslist.json',
    'val': '/home/joel-zhang/file/tdiuc/ban_ans/val_anslist.json'
}
TARGETPTH = {
    'train': '/home/joel-zhang/file/tdiuc/cache/train_target.pkl',
    'val': '/home/joel-zhang/file/tdiuc/cache/val_target.pkl'
}


class tdiuc_Dataset(Dataset):
    def __init__(self, name, maxlen):
        
        # questions
        quesdata = json.load(open(QUESPATH[name]))['questions']

        # images
        print('loading features from h5 file')
        with h5py.File(IMGPATH[name], 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.img_id2idx = cPickle.load(open(IMGIDPATH[name], 'rb'))

        # answer
        ans_list = json.load(open(ANSPTH[name]))
        self.quesid2ans = {}
        for datum in ans_list:
            flag = 1
            for k in datum['answer'].keys():
                if len(k) == 0:
                    flag = 0
                    break
            if flag:
                self.quesid2ans[datum['question_id']] = datum['answer']

        # target
        targets = cPickle.load(open(TARGETPTH[name], 'rb'))
        self.label = {}
        for datum in targets:
            self.label[datum['question_id']] = datum['labels']

        self.queslist = []
        for datum in quesdata:
            if len(self.label[datum['question_id']]) > 0:
                self.queslist.append(datum)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        print("%s: Use %d data in dataset\n" % (name, len(self.queslist)))

        self.maxlen = maxlen

    def __len__(self):
        return len(self.queslist)

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
        datum = self.queslist[item]

        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']

        # question
        ques_ids = self.proc_query(ques, self.maxlen[0])

        # answer
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

        # image
        idx = self.img_id2idx[img_id]
        features = self.features[self.pos_boxes[idx][0]:self.pos_boxes[idx][1], :]
        features = features[:50]
        features = torch.from_numpy(features)
        
        return ques_id, ques_ids, answer, features.float(), score.float()



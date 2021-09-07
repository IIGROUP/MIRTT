from __future__ import print_function
import os
import json
import h5py
import _pickle as cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch._six import string_classes
import collections
import torch.nn.functional as F


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
TARGETPTH = {
    'train': '/home/joel-zhang/file/tdiuc/cache/train_target.pkl',
    'val': '/home/joel-zhang/file/tdiuc/cache/val_target.pkl'
}


class tdiuc_Dataset(Dataset):
    def __init__(self, name, maxlen):
        
        # questions
        self.queslist = json.load(open(QUESPATH[name]))['questions']

        # images
        print('loading features from h5 file')
        with h5py.File(IMGPATH[name], 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.img_id2idx = cPickle.load(open(IMGIDPATH[name], 'rb'))

        # target
        targets = cPickle.load(open(TARGETPTH[name], 'rb'))
        self.label = {}
        for datum in targets:
            self.label[datum['question_id']] = datum['labels']

        self.ans2label = cPickle.load(open('/home/joel-zhang/file/tdiuc/cache/trainval_ans2label.pkl', 'rb'))
        self.label2ans = cPickle.load(open('/home/joel-zhang/file/tdiuc/cache/trainval_label2ans.pkl', 'rb'))
        self.num_ans_candidates = len(self.ans2label)

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
        ques_ids = self.proc_query(ques, self.maxlen)

        # image
        idx = self.img_id2idx[img_id]
        features = self.features[self.pos_boxes[idx][0]:self.pos_boxes[idx][1], :]
        features = features[:50]
        features = torch.from_numpy(features)

        # label
        label = self.label[ques_id]
        target = torch.zeros(self.num_ans_candidates)
        if label is not None:
            target[label] = 1

        return ques_id, features.float(), ques_ids, target


# compute score
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    model.eval()
    with torch.no_grad():
        for (ques_id, feats, ques, target) in iter(dataloader):
            feats, ques, target = feats.cuda(), ques.cuda(), target.cuda()
            pred = model(feats, ques)

            batch_score = compute_score_with_logits(pred, target).sum()
            score += batch_score
            upper_bound += (target.max(1)[0]).sum()

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
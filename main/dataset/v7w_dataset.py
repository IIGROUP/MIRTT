import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import json
import _pickle as cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import h5py
import itertools
from transformers import BertTokenizer
from torch._six import string_classes
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


def _create_entry(img, question, answer, label, ans_gt, ans_mc):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'label'      : label,
        'ans_gt'      : ans_gt,
        'ans_mc'      : ans_mc}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans, ans_candidates):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    question_path = os.path.join(
        dataroot, 'v7w_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    entries = []
    for question in questions:
        img_id = question['image_id']
        ans_mc = ans_candidates[str(question['question_id'])]['mc']
        ans_gt = ans_candidates[str(question['question_id'])]['ans_gt']
        label = ans_candidates[str(question['question_id'])]['label']

        entries.append(_create_entry(img_id2val[img_id], question, None, label, ans_gt, ans_mc))

    return entries


class v7wDataset(Dataset):
    def __init__(self, name, dataroot='/home/joel-zhang/file/ICCV19_VQA-CTI/data_v7w', max_boxes=100, maxlen=[12, 6], adaptive=True):
        super(v7wDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        answer_candidate_path = os.path.join(dataroot, 'answer_%s.json' % name)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.answer_candidates = json.load(open(answer_candidate_path))
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.adaptive = adaptive
        self.max_box = max_boxes
        self.maxlen = maxlen
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))
        
        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        print('loading features from h5 file. Path:', str(h5_path))
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans, self.answer_candidates)

        self.tensorize()
        self.FindType()

    def TextEncode(self, text, length):
        word_pieces = ["[CLS]"]
        tokens_text = self.tokenizer.tokenize(text)
        word_pieces += tokens_text + ["[SEP]"]

        if len(word_pieces) < length:
            word_pieces = word_pieces + ['[PAD]' for _ in range(length - len(word_pieces))] #Padding sentences
        else:
            word_pieces = word_pieces[:length-1] + ['[SEP]'] #Prunning the list to be of specified max length
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        padding = [0] * (length - len(ids))
        ids += padding
        tokens_tensor = torch.tensor(ids)

        return tokens_tensor

    def tensorize(self):
        self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            qst_ids = self.TextEncode(entry['question'], self.maxlen[0])
            entry['q_token'] = qst_ids

            tokens_0 = self.TextEncode(entry['ans_mc'][0], self.maxlen[1])
            tokens_1 = self.TextEncode(entry['ans_mc'][1], self.maxlen[1])
            tokens_2 = self.TextEncode(entry['ans_mc'][2], self.maxlen[1])
            tokens_3 = self.TextEncode(entry['ans_mc'][3], self.maxlen[1])
            ans_ids = torch.stack([tokens_0, tokens_1, tokens_2, tokens_3])

            entry['ans_mc_token'] = ans_ids

            label = torch.from_numpy(np.float32(entry['label']))
            entry['label'] = label

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None
    
    def FindType(self):
        qst_type = ['What', 'Where', 'When', 'Who', 'Why', 'How'] # 0, 1, 3, 4, 5
        print('question type:', qst_type)
        for entry in self.entries:
            for i, w in enumerate(qst_type):
                if w in entry['question']:
                    entry['qst_type'] = i

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

            features = features[:self.max_box]

        q_id = entry['question_id']
        question = entry['q_token']
        ans_mc = entry['ans_mc_token']
        label = entry['label']

        return q_id, question, ans_mc, features, label

    def __len__(self):
        return len(self.entries)

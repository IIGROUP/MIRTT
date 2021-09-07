import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import random
import os
import json
import csv
import sys
import base64
import time
import psutil
from torch._six import string_classes
import collections
import _pickle as cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import h5py
from transformers import BertPreTrainedModel, BertConfig, BertModel, BertTokenizer

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

# ------------------------------
# masked language
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def random_word(tokens):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = 0.15
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def textmaskencode(text, max_seq_length):
    tokens = tokenizer.tokenize(text.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(tokens)

    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    seq_len = len(masked_tokens)
    if len(masked_tokens) < max_seq_length:
        masked_tokens += ['[PAD]' for _ in range(max_seq_length - len(masked_tokens))] #Padding sentences

    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [0] * seq_len

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
    while len(input_mask) < max_seq_length:
        input_mask.append(1)
    while len(lm_label_ids) < max_seq_length:
        lm_label_ids.append(-1)

    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(lm_label_ids)


# ------------------------------
# ------- v7w dataset ----------
# ------------------------------
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
        self.features = torch.from_numpy(self.features)     

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            features = features[:self.max_box]

        question, q_mask, q_lm_label_ids = textmaskencode(entry['question'], self.maxlen[0])
        
        ans_gt = entry['ans_gt']

        ans, a_mask, a_lm_label_ids = textmaskencode(ans_gt, self.maxlen[1])

        return question, ans, features, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids

    def __len__(self):
        return len(self.entries)


# ------------------------------
# ------ TDIUC dataset ---------
# ------------------------------
def _create_td_entry(img, question, answer, ans):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'ans'         : ans}
    return entry

def _load_TDIUC_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'TDIUC_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    assert len(questions) == len(answers)
    entries = []

    for question, answer in zip(questions, answers):
        assert question['question_id'] == answer['question_id']
        assert question['image_id'] == answer['image_id']
        img_id = question['image_id']
        if len(answer['labels']) > 0 and answer['labels'][0] != 1 and answer['labels'][0] != 1218:
            entries.append(_create_td_entry(img_id2val[img_id], question, None, answer['labels']))

    return entries

class TDIUCDataset(Dataset):
    def __init__(self, name, dataroot='/home/joel-zhang/file/tdiuc', max_boxes=100, maxlen=[12, 6], adaptive=True):
        super(TDIUCDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')

        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
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

        self.entries = _load_TDIUC_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        self.features = torch.from_numpy(self.features)  

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            features = features[:self.max_box]

        question, q_mask, q_lm_label_ids = textmaskencode(entry['question'], self.maxlen[0])
        label = entry['ans']
        ans = self.label2ans[label[0]]
        ans, a_mask, a_lm_label_ids = textmaskencode(ans, self.maxlen[1])
        return question, ans, features, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids

    def __len__(self):
        return len(self.entries)


# ------------------------------
# ------ vqa2.0 dataset --------
# ------------------------------
# extract region feature
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

class VQADataset(Dataset):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        # quesitons
        queslist = \
            json.load(open('/home/joel-zhang/file/lxmert/data/vqa/train.json', 'r')) + \
            json.load(open('/home/joel-zhang/file/lxmert/data/vqa/nominival.json', 'r')) + \
            json.load(open('/home/joel-zhang/file/lxmert/data/vqa/minival.json', 'r')) + \
            json.load(open('/home/joel-zhang/file/lxmert/data/vqa/vg.json', 'r'))
        # images
        self.imgdata = {}
        train_feat_path = '/home/joel-zhang/file/lxmert/data/vqa/img_feat/train2014_obj36.tsv'
        val_feat_path = '/home/joel-zhang/file/lxmert/data/vqa/img_feat/val2014_obj36.tsv'
        self.imgdata.update(load_obj_tsv(train_feat_path))
        self.imgdata.update(load_obj_tsv(val_feat_path))
        # Only keep the data with loaded image features
        self.quesdata = []  # question information
        for datum in queslist:
            label = datum['label']
            best_score = 0
            right_ans = ''
            for k, v in label.items():
                if v > best_score:
                    best_score = v
                    right_ans = k
            if len(right_ans) == 0:
                continue
            if datum['img_id'] in self.imgdata or 'COCO_train2014_000000'+str(datum['img_id']) in self.imgdata or 'COCO_val2014_000000'+str(datum['img_id']) in self.imgdata:
                datum['answer'] = right_ans
                self.quesdata.append(datum)
        
    def __len__(self):
        return len(self.quesdata)

    def __getitem__(self, index):
        datum = self.quesdata[index]

        img_id = datum['img_id']
        question, q_mask, q_lm_label_ids = textmaskencode(datum['sent'], self.maxlen[0])
        ans, a_mask, a_lm_label_ids = textmaskencode(datum['answer'], self.maxlen[1])

        if isinstance(img_id, int): # from vg
            if 'COCO_train2014_000000'+str(img_id) in self.imgdata:
                feats = self.imgdata['COCO_train2014_000000'+str(img_id)]
            else:
                feats = self.imgdata['COCO_val2014_000000'+str(img_id)]
        else:
            feats = self.imgdata[img_id]
        feats = torch.from_numpy(feats).float()
        
        return question, ans, feats, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids
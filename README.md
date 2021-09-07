# MIRTT

This repository is the implementation of **MIRTT: Learning Multimodal Interaction Representations from Trilinear Transformers for Visual Question Answering**.

## Data Source

VQA2.0, GQA: [LXMERT](https://github.com/airsplay/lxmert)

visual7w, TDIUC: [CTI](https://github.com/aioz-ai/ICCV19_VQA-CTI)

VQA1.0: [VQA web](https://visualqa.org/vqa_v1_download.html)

## Pretrain

Under ./pretrain:

> bash run.bash exp_name gpuid

Some parameters can be changed in run.bash.

## MC VQA

Under ./main:

> bash run.bash exp_name gpuid

## FFOE VQA

Two stage workflow

Stage one: bilinear model (BAN, SAN, MLP)

Under ./bilinear_method:

> bash run.bash exp_name gpuid mod dataset model

After training, we can generate answer list for each dataset. In this way, we simplify FFOE VQA into MC VQA.

Stage two: MIRTT. Under ./main

----
keep updating
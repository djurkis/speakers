#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
from datetime import datetime

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    SentencesDataset,
    InputExample,
    losses,
    SentenceTransformer,
)

from logzero import logger


class_map = {"person1": 0, "person2": 1, "person3": 2, "person4": 3, "note": 4}


def make_examples(path):

    with open(path) as f:
        data = json.load(f)

    # contrastive
    contrastive = []

    mulneg = []
    # MultipleNegative
    # https://www.sbert.net/examples/training/quora_duplicate_questions/README.html#multiplenegativesrankingloss

    # maybe smooth out the simmiliarity label for the same person over time to
    # consider flow of topic in to the labels.
    for dial in data:
        sents = [line["line"] for line in dial]
        labels = [class_map[line["tag"]] for line in dial]

        for s, l in zip(sents, labels):
            for s2, l2 in zip(sents[1:], labels[1:]):
                if l2 == l:
                    mulneg.append(InputExample(texts=[s, s2], label=1.0))
                    mulneg.append(InputExample(texts=[s2, s], label=1.0))
                    # symetry

                    contrastive.append(InputExample(texts=[s, s2], label=0))

                if l != l2:
                    contrastive.append(InputExample(texts=[s, s2], label=1))

    return contrastive, mulneg


def make_sents(path):
    with open(path) as f:
        data = json.load(f)

    # contrastive
    se1,se2 = [],[]
    labels = []

    # MultipleNegative
    # https://www.sbert.net/examples/training/quora_duplicate_questions/README.html#multiplenegativesrankingloss

    # maybe smooth out the simmiliarity label for the same person over time to
    # consider flow of topic in to the labels.
    for dial in data:
        sents = [line["line"] for line in dial]
        labels = [class_map[line["tag"]] for line in dial]

        for s, l in zip(sents, labels):
            for s2, l2 in zip(sents[1:], labels[1:]):
                if l2 == l:
                    se1.append(s)
                    se2.append(s2)
                if l != l2:
                    se1.append(s)
                    se2.append(s2)
                    labels.append(0)

    return se1,se2,labels



#
# class SimDataset(DataSet):
#     def __init__(self):
#         pass


def main(args):

    con, mul = make_examples(args.train_path)
    s1,s2,l = make_examples(args.dev_path)

    eval = sentence_transformers.evaluation.BinaryClassificationEvaluator(
    s1,s2,l,name="output_dev"
    )

    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

    train_mulneg_dataset = SentencesDataset(mul, model=model)
    train_mulneg_dataloader = DataLoader(
        train_mulneg_dataset, shuffle=True, batch_size=args.batch_size
    )
    train_mulneg_loss = losses.MultipleNegativesRankingLoss(model)

    train_contrastive_dataset = SentencesDataset(con, model=model)
    train_contrastive_dataloader = DataLoader(
        train_contrastive_dataset, shuffle=True, batch_size=args.batch_size
    )
    train_contrastive_loss = losses.OnlineContrastiveLoss(
        model, distance_metric=distance_metric, margin=0.5
    )

    model_save_path = (
        args.out_dir + "training_multi" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    os.makedirs(model_save_path, exist_ok=True)

    model.fit(
        train_objectives=[
            (train_mulneg_dataloader, train_mulneg_loss),
            (train_contrastive_dataloader, train_contrastive_loss),
        ],
        epochs=args.epochs,
        warmup_steps=1000,
        output_path="out",
    )
    model.save("output2_big")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_path",
        type=str,
        default="/home/jurkis/zima2020/THEaiTRE/dataset/10movies_max4_tag.eval.json",
    )
    ap.add_argument("--dev_path", type=str)

    ap.add_argument("--eval_path", type=str)

    ap.add_argument("--out_dir", type=str, default="training")
    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap = ap.parse_args()
    main(ap)

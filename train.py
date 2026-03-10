"""
train.py — FICount
Training script for FICount: Prototype-Guided Attention Matching for Few-Shot Insect Counting

Adapted from: Learning To Count Everything, CVPR 2021
              Viresh Ranjan, Udbhav, Thu Nguyen, Minh Hoai
"""

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from os.path import exists, join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

from model import FICount, PVGDiscriminator, weights_normal_init
from utils import (
    TransformTrain, Transform,
    exemplar_count_loss, wgan_gp_loss, cosine_identity_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="FICount Training")
parser.add_argument("-dp",  "--data_path",      type=str,   default='./data/',
                    help="Path to dataset root")
parser.add_argument("-o",   "--output_dir",     type=str,   default="./logs",
                    help="Directory for checkpoints and stats")
parser.add_argument("-ts",  "--test_split",     type=str,   default='val',
                    choices=["train", "test", "val"])
parser.add_argument("-ep",  "--epochs",         type=int,   default=1500)
parser.add_argument("-g",   "--gpu",            type=int,   default=0)
parser.add_argument("-lr",  "--learning_rate",  type=float, default=1e-4)
parser.add_argument("-dlr", "--disc_lr",        type=float, default=1e-4,
                    help="Learning rate for PVG discriminators")
parser.add_argument("-lx",  "--lambda_ex",      type=float, default=1.0,
                    help="Weight for exemplar count loss")
parser.add_argument("-la",  "--lambda_adv",     type=float, default=0.1,
                    help="Weight for PVG adversarial loss")
parser.add_argument("-li",  "--lambda_id",      type=float, default=1.0,
                    help="Weight for cosine identity loss")
parser.add_argument("-M",   "--num_variants",   type=int,   default=2,
                    help="Number of PVG variants per exemplar")
parser.add_argument("-k",   "--num_shots",      type=int,   default=4,
                    choices=[1, 2, 3, 4],
                    help="Number of exemplar shots (boxes used per image)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(args.output_dir, exist_ok=True)
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

anno_file       = join(args.data_path, 'annotation_FSC147_384.json')
data_split_file = join(args.data_path, 'Train_Test_Val_FSC_147.json')
im_dir          = join(args.data_path, 'images_384_VarV2')
gt_dir          = join(args.data_path, 'gt_density_map_adaptive_384_VarV2')

with open(anno_file)       as f: annotations = json.load(f)
with open(data_split_file) as f: data_split  = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

model = FICount(M=args.num_variants).to(device)
weights_normal_init(model.pvg3,  dev=0.001)
weights_normal_init(model.pvg4,  dev=0.001)
weights_normal_init(model.lawc3, dev=0.001)
weights_normal_init(model.lawc4, dev=0.001)
weights_normal_init(model.decoder, dev=0.001)
model.train()

# discriminators for PVG adversarial training
disc3 = PVGDiscriminator(channels=1024).to(device)
disc4 = PVGDiscriminator(channels=2048).to(device)

# optimisers — backbone is frozen; discriminators have separate optimiser
optimizer      = optim.Adam(
    list(model.pvg3.parameters())   + list(model.pvg4.parameters())  +
    list(model.lawc3.parameters())  + list(model.lawc4.parameters()) +
    list(model.decoder.parameters()),
    lr=args.learning_rate,
)
optimizer_disc = optim.Adam(
    list(disc3.parameters()) + list(disc4.parameters()),
    lr=args.disc_lr,
)

criterion = nn.MSELoss().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    model.train()
    disc3.train()
    disc4.train()

    im_ids = data_split['train']
    random.shuffle(im_ids)

    train_mae  = 0.0
    train_rmse = 0.0
    train_loss = 0.0
    cnt        = 0
    pbar       = tqdm(im_ids)

    for im_id in pbar:
        cnt += 1
        anno  = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots   = np.array(anno['points'])

        rects = []
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])
        # apply shot limit
        rects = rects[:args.num_shots]

        image        = Image.open(join(im_dir, im_id))
        image.load()
        density_path = join(gt_dir, im_id.split(".jpg")[0] + ".npy")
        density      = np.load(density_path).astype('float32')

        sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
        sample = TransformTrain(sample)
        image_t   = sample['image'].to(device)        # [3, H, W]
        boxes_t   = sample['boxes'].to(device)        # [K, 4]
        gt_density = sample['gt_density'].to(device)  # [1, 1, H, W]

        image_t   = image_t.unsqueeze(0)              # [1, 3, H, W]
        boxes_list = [boxes_t]                         # list of length B=1

        # ── forward (training mode returns PVG fakes) ────────────────────
        D_hat, D_raw, R, extras = model(image_t, boxes_list, training=True)

        # align gt to output size if needed
        if D_hat.shape[-2:] != gt_density.shape[-2:]:
            orig_cnt = gt_density.sum().item()
            gt_density = F.interpolate(gt_density, size=D_hat.shape[-2:], mode='bilinear', align_corners=False)
            new_cnt = gt_density.sum().item()
            if new_cnt > 0:
                gt_density = gt_density * (orig_cnt / new_cnt)

        # ── density loss ─────────────────────────────────────────────────
        L_density = criterion(D_hat, gt_density)

        # ── exemplar count loss ──────────────────────────────────────────
        L_ex = exemplar_count_loss(D_hat, boxes_list)

        # ── PVG adversarial + identity losses ────────────────────────────
        fakes3 = extras['pvg_fakes3']    # [K*M, C3, p, p]
        fakes4 = extras['pvg_fakes4']
        reals3 = extras['exemplars3']    # [K,   C3, p, p]
        reals4 = extras['exemplars4']

        # repeat reals M times to match fakes count
        K  = reals3.shape[0]
        M  = args.num_variants
        reals3_rep = reals3.repeat_interleave(M, dim=0)   # [K*M, C3, p, p]
        reals4_rep = reals4.repeat_interleave(M, dim=0)

        # discriminator step
        optimizer_disc.zero_grad()
        d_loss3, g_loss3, _ = wgan_gp_loss(disc3, reals3, fakes3)
        d_loss4, g_loss4, _ = wgan_gp_loss(disc4, reals4, fakes4)
        L_disc = d_loss3 + d_loss4
        L_disc.backward(retain_graph=True)
        optimizer_disc.step()

        # generator (model) step
        L_adv = g_loss3 + g_loss4
        L_id  = (cosine_identity_loss(fakes3, reals3_rep)
                 + cosine_identity_loss(fakes4, reals4_rep))

        L_total = (L_density
                   + args.lambda_ex  * L_ex
                   + args.lambda_adv * L_adv
                   + args.lambda_id  * L_id)

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        # ── metrics ──────────────────────────────────────────────────────
        train_loss += L_total.item()
        pred_cnt    = D_hat.sum().item()
        gt_cnt      = gt_density.sum().item()
        cnt_err     = abs(pred_cnt - gt_cnt)
        train_mae  += cnt_err
        train_rmse += cnt_err ** 2

        pbar.set_description(
            'actual-predicted: {:6.1f}, {:6.1f}, err: {:6.1f}. '
            'MAE: {:5.2f} RMSE: {:5.2f} | Best val MAE: {:5.2f} RMSE: {:5.2f}'.format(
                gt_cnt, pred_cnt, cnt_err,
                train_mae / cnt, (train_rmse / cnt) ** 0.5,
                best_mae, best_rmse,
            )
        )
        print("")

    n = len(im_ids)
    return train_loss / n, train_mae / n, (train_rmse / n) ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def eval_split(split_name):
    model.eval()
    cnt = SAE = SSE = 0

    print("Evaluation on {} data".format(split_name))
    im_ids = data_split[split_name]
    pbar   = tqdm(im_ids)

    for im_id in pbar:
        anno   = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots   = np.array(anno['points'])

        rects = []
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])
        rects = rects[:args.num_shots]

        image = Image.open(join(im_dir, im_id))
        image.load()
        sample = {'image': image, 'lines_boxes': rects}
        sample = Transform(sample)
        image_t   = sample['image'].to(device).unsqueeze(0)
        boxes_t   = sample['boxes'].to(device)
        boxes_list = [boxes_t]

        with torch.no_grad():
            D_hat, _, _, _ = model(image_t, boxes_list, training=False)

        gt_cnt   = dots.shape[0]
        pred_cnt = D_hat.sum().item()
        cnt     += 1
        err      = abs(gt_cnt - pred_cnt)
        SAE     += err
        SSE     += err ** 2

        pbar.set_description(
            '{:<8}: actual-predicted: {:6d}, {:6.1f}, err: {:6.1f}. '
            'MAE: {:5.2f} RMSE: {:5.2f}'.format(
                im_id, gt_cnt, pred_cnt, err,
                SAE / cnt, (SSE / cnt) ** 0.5,
            )
        )
        print("")

    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(
        split_name, SAE / cnt, (SSE / cnt) ** 0.5))
    return SAE / cnt, (SSE / cnt) ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

best_mae  = 1e7
best_rmse = 1e7
stats     = []

for epoch in range(args.epochs):
    model.train()
    train_loss, train_mae, train_rmse = train()

    model.eval()
    val_mae, val_rmse = eval_split(args.test_split)

    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))

    stats_file = join(args.output_dir, "stats.txt")
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))

    if val_mae <= best_mae:
        best_mae  = val_mae
        best_rmse = val_rmse
        model_name = join(args.output_dir, "FICount_best.pth")
        torch.save(model.state_dict(), model_name)

    print(
        "Epoch {:4d} | Loss: {:.4f} | Train MAE: {:.2f} RMSE: {:.2f} "
        "| Val MAE: {:.2f} RMSE: {:.2f} | Best Val MAE: {:.2f} RMSE: {:.2f}".format(
            epoch + 1, stats[-1][0],
            stats[-1][1], stats[-1][2],
            stats[-1][3], stats[-1][4],
            best_mae, best_rmse,
        )
    )

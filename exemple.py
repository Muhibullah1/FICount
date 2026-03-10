"""
demo.py — FICount
Demo script for FICount: Prototype-Guided Attention Matching for Few-Shot Insect Counting

Usage:
  # draw exemplar boxes interactively
  python demo.py -i /path/to/image.jpg -m FICount_best.pth -o ./output

  # supply exemplar boxes from a file (one box per line: y1 x1 y2 x2)
  python demo.py -i /path/to/image.jpg -b boxes.txt -m FICount_best.pth -o ./output

Adapted from: Few Shot Counting Demo, Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
"""

import os
import argparse
import torch
import cv2
from PIL import Image

from model import FICount
from utils import Transform, visualize_output_and_save, select_exemplar_rois


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="FICount Demo")
parser.add_argument("-i", "--input_image",  type=str, required=True,
                    help="Path to input image")
parser.add_argument("-b", "--bbox_file",    type=str, default=None,
                    help="Path to bounding-box file (y1 x1 y2 x2 per line). "
                         "If omitted, interactive selection is used.")
parser.add_argument("-o", "--output_dir",   type=str, default=".",
                    help="Directory to save the output visualisation")
parser.add_argument("-m", "--model_path",   type=str, required=True,
                    help="Path to trained FICount checkpoint (.pth)")
parser.add_argument("-g", "--gpu_id",       type=int, default=0,
                    help="GPU id. Use -1 for CPU.")
parser.add_argument("-M", "--num_variants", type=int, default=2,
                    help="Number of PVG variants (must match training setting)")
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    device  = torch.device('cpu')
    print("===> Using CPU mode.")
else:
    use_gpu = True
    device  = torch.device('cuda')
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

os.makedirs(args.output_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = FICount(M=args.num_variants).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Exemplar boxes
# ─────────────────────────────────────────────────────────────────────────────

image_name = os.path.splitext(os.path.basename(args.input_image))[0]

if args.bbox_file is None:
    # interactive selection
    out_bbox_file = join(args.output_dir, "{}_boxes.txt".format(image_name))
    fout = open(out_bbox_file, "w")

    im = cv2.imread(args.input_image)
    cv2.imshow('image', im)
    rects = select_exemplar_rois(im)

    for r in rects:
        fout.write("{} {} {} {}\n".format(r[0], r[1], r[2], r[3]))
    fout.close()
    cv2.destroyWindow("image")
    print("Selected boxes saved to {}".format(out_bbox_file))
else:
    rects = []
    with open(args.bbox_file, "r") as fin:
        for line in fin:
            data = line.split()
            y1, x1, y2, x2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            rects.append([y1, x1, y2, x2])

print("Bounding boxes: ", rects)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

image = Image.open(args.input_image)
image.load()
sample    = {'image': image, 'lines_boxes': rects}
sample    = Transform(sample)
image_t   = sample['image'].to(device).unsqueeze(0)   # [1, 3, H, W]
boxes_t   = sample['boxes'].to(device)                 # [K, 4]
boxes_list = [boxes_t]

with torch.no_grad():
    D_hat, _, _, _ = model(image_t, boxes_list, training=False)

print('===> Predicted count: {:6.2f}'.format(D_hat.sum().item()))

# ─────────────────────────────────────────────────────────────────────────────
# Save visualisation
# ─────────────────────────────────────────────────────────────────────────────

rslt_file = join(args.output_dir, "{}_out.png".format(image_name))
visualize_output_and_save(
    image_t.squeeze(0).detach().cpu(),
    D_hat.detach().cpu(),
    boxes_t.cpu(),
    rslt_file,
)
print("===> Visualisation saved to {}".format(rslt_file))


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def join(*args):
    return os.path.join(*args)

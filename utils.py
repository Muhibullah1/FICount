"""
utils.py — FICount
Keeps original image-transform helpers and visualisation utilities.
Replaces extract_features / MincountLoss / PerturbationLoss with
FICount-specific losses: exemplar_count_loss, wgan_gp_loss, cosine_identity_loss.
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

matplotlib.use('agg')


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_HW       = 384
MAX_HW       = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD  = [0.229, 0.224, 0.225]

Normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Image / box transforms  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class resizeImage(object):
    """Resize so the longest side ≤ MAX_HW and both dims are divisible by 8.
    Aspect ratio is preserved.  No resize if already within bounds.
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes = sample['image'], sample['lines_boxes']

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1

        boxes = []
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([y1, x1, y2, x2])

        image = Normalize(image)
        sample = {'image': image, 'boxes': torch.Tensor(boxes)}
        return sample


class resizeImageWithGT(object):
    """Same as resizeImage but also resizes the ground-truth density map,
    preserving the total count sum.
    By: Minh Hoai Nguyen.  Modified by: Viresh.
    """

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes, density = (
            sample['image'], sample['lines_boxes'], sample['gt_density']
        )

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            image           = transforms.Resize((new_H, new_W))(image)
            orig_count      = np.sum(density)
            density         = cv2.resize(density, (new_W, new_H))
            new_count       = np.sum(density)
            if new_count > 0:
                density = density * (orig_count / new_count)
        else:
            scale_factor = 1

        boxes = []
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([y1, x1, y2, x2])

        image   = Normalize(image)
        density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0).float()
        sample  = {'image': image, 'boxes': torch.Tensor(boxes), 'gt_density': density}
        return sample


Transform      = transforms.Compose([resizeImage(MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


# ─────────────────────────────────────────────────────────────────────────────
# Density map construction
# ─────────────────────────────────────────────────────────────────────────────

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """2D Gaussian kernel matching MATLAB's fspecial('gaussian', ...).
    By: Minh Hoai Nguyen.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h    = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# ─────────────────────────────────────────────────────────────────────────────
# FICount losses
# ─────────────────────────────────────────────────────────────────────────────

def exemplar_count_loss(output, boxes_list):
    """L_ex: penalise deviation from an integral of 1 inside each exemplar box.

    Each exemplar bounding box should contain exactly one insect (the annotated
    instance), so the density sum inside the box is encouraged to equal 1.

    Args:
        output     : [B, 1, H, W]  predicted density map
        boxes_list : list of B tensors, each [K, 4]  [y1,x1,y2,x2] in image coords

    Returns:
        scalar loss (mean over B·K)
    """
    loss  = 0.0
    count = 0
    B, _, H, W = output.shape

    for b in range(B):
        boxes = boxes_list[b]   # [K, 4]
        for k in range(boxes.shape[0]):
            y1 = max(int(boxes[k, 0]), 0)
            x1 = max(int(boxes[k, 1]), 0)
            y2 = min(int(boxes[k, 2]), H)
            x2 = min(int(boxes[k, 3]), W)
            if y2 > y1 and x2 > x1:
                region_sum = output[b, 0, y1:y2, x1:x2].sum()
                loss  += (region_sum - 1.0) ** 2
                count += 1

    return loss / max(count, 1)


def wgan_gp_loss(discriminator, real_feats, fake_feats, lambda_gp=10.0):
    """WGAN-GP adversarial loss for the PVG discriminator.

    Args:
        discriminator : PVGDiscriminator
        real_feats    : [N, C, h, w]  real exemplar feature maps
        fake_feats    : [N, C, h, w]  PVG-generated feature maps
        lambda_gp     : gradient penalty weight

    Returns:
        d_loss : discriminator loss  (minimise to train D)
        g_loss : generator loss      (minimise to train G)
        gp     : gradient penalty scalar (for logging)
    """
    d_real = discriminator(real_feats).mean()
    d_fake = discriminator(fake_feats.detach()).mean()

    # gradient penalty
    N     = min(real_feats.shape[0], fake_feats.shape[0])
    alpha = torch.rand(N, 1, 1, 1, device=real_feats.device)
    interp = (alpha * real_feats[:N]
              + (1 - alpha) * fake_feats[:N].detach()).requires_grad_(True)
    d_interp = discriminator(interp)
    grads    = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    gp = lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()

    d_loss = d_fake - d_real + gp          # D maximises real - fake  ↔  minimise fake - real
    g_loss = -discriminator(fake_feats).mean()   # G maximises D(fake)

    return d_loss, g_loss, gp


def cosine_identity_loss(fake_feats, real_feats):
    """L_id: cosine distance between GAP of each fake and its conditioning real.

    Keeps each PVG variant semantically anchored to the exemplar it was
    conditioned on, preventing mode collapse toward different insect classes.

    Args:
        fake_feats : [N, C, h, w]  — N = K × M fakes
        real_feats : [N, C, h, w]  — matching real exemplars (repeated M times)

    Returns:
        scalar in [0, 2]
    """
    fv = F.adaptive_avg_pool2d(fake_feats, 1).view(fake_feats.shape[0], -1)
    rv = F.adaptive_avg_pool2d(real_feats, 1).view(real_feats.shape[0], -1)
    return (1.0 - F.cosine_similarity(fv, rv, dim=1)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation  (unchanged from original, adapted box format [y1,x1,y2,x2])
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverse ImageNet normalisation for display.  Input/output: [C,H,W]."""
    denormalized = tensor.clone()
    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized


def format_for_plotting(tensor):
    """Reshape [N,C,H,W] or [C,H,W] → [H,W,C] or [H,W] for imshow."""
    has_batch = len(tensor.shape) == 4
    formatted  = tensor.clone()
    if has_batch:
        formatted = formatted.squeeze(0)
    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    return formatted.permute(1, 2, 0).detach()


def visualize_output_and_save(input_, output, boxes, save_path,
                               figsize=(20, 12), dots=None):
    """Four-panel visualisation: input · overlay · density · density+boxes.

    boxes: [K, 4]  [y1, x1, y2, x2] (no batch-index prefix)
    dots : Nx2 numpy array of GT dot locations, or None
    """
    pred_cnt = output.sum().item()

    # ensure boxes is 2-D [K, 4]
    if boxes.dim() == 3:
        boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(boxes.shape[0]):
        y1 = int(boxes[i, 0].item())
        x1 = int(boxes[i, 1].item())
        y2 = int(boxes[i, 2].item())
        x2 = int(boxes[i, 3].item())
        roi_cnt = output[0, 0, y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1   = format_for_plotting(denormalize(input_))
    out_np = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # ── panel 1: input image with exemplar boxes ──────────────────────────
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)
    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[:4]
        rect  = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1, edgecolor='k',
                                   linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    # ── panel 2: overlay ──────────────────────────────────────────────────
    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))
    gray = (0.2989 * img1[:, :, 0]
            + 0.5870 * img1[:, :, 1]
            + 0.1140 * img1[:, :, 2])
    ax.imshow(gray, cmap='gray')
    ax.imshow(out_np, cmap=plt.cm.viridis, alpha=0.5)

    # ── panel 3: density map ──────────────────────────────────────────────
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(out_np)

    # ── panel 4: density map + exemplar boxes with per-box counts ─────────
    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map with exemplars")
    ret_fig = ax.imshow(out_np)
    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox
        rect  = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1, edgecolor='k',
                                   linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')
    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def select_exemplar_rois(image):
    """Interactive exemplar selection via OpenCV ROI selector.
    Press 'n' + drag to draw a box.  Press 'q' or Esc to finish.
    Returns list of [y1, x1, y2, x2] boxes.
    """
    all_rois = []
    print("Press 'q' or Esc to quit. "
          "Press 'n' then drag to draw an exemplar box.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == ord('\r'):
            rect = cv2.selectROI("image", image, False, False)
            x1   = rect[0]
            y1   = rect[1]
            x2   = x1 + rect[2] - 1
            y2   = y1 + rect[3] - 1
            all_rois.append([y1, x1, y2, x2])
            for r in all_rois:
                ry1, rx1, ry2, rx2 = r
                cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            print("Press 'q' or Esc to quit. "
                  "Press 'n' then drag to draw another exemplar box.")
    return all_rois


def scale_and_clip(val, scale_factor, min_val, max_val):
    """Helper to scale a value and clip within range."""
    new_val = int(round(val * scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val

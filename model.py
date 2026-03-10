"""
model.py — FICount
Backbone : frozen ResNet-50, layer3 (stride-16, C=1024) and layer4 (stride-32, C=2048)
Components: PVG · PVGDiscriminator · LAWC · DensityDecoder (+ CGDM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────

#-------------------Model.py-------------------------
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# PVG — Prototype Variation Generator
# ─────────────────────────────────────────────────────────────────────────────

class PVG(nn.Module):
    """Generate M appearance variants of an exemplar feature map.

    Generator G: three 1×1 conv layers (GroupNorm + ReLU).
    Noise vector z_m ~ N(0,I) is projected to channel space and added to the
    exemplar before generation, driving per-variant diversity.

    Args:
        channels  : C_ell  (1024 for layer3, 2048 for layer4)
        noise_dim : dimension of the Gaussian noise vector
        M         : number of variants to produce per exemplar
    """

    def __init__(self, channels: int, noise_dim: int = 256, M: int = 2):
        super().__init__()
        self.M         = M
        self.noise_dim = noise_dim
        self.noise_proj = nn.Linear(noise_dim, channels)
        self.gen = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, exemplar: torch.Tensor):
        """
        exemplar : [E, C, h, w]
        returns  : [E, M, C, h, w]
        """
        E, C, h, w = exemplar.shape
        device = exemplar.device
        variants = []
        for _ in range(self.M):
            z      = torch.randn(E, self.noise_dim, device=device)
            z_proj = self.noise_proj(z).view(E, C, 1, 1)   # broadcast over h,w
            variants.append(self.gen(exemplar + z_proj))    # [E, C, h, w]
        return torch.stack(variants, dim=1)                 # [E, M, C, h, w]


# ─────────────────────────────────────────────────────────────────────────────
# PVG Discriminator  (training only)
# ─────────────────────────────────────────────────────────────────────────────

class PVGDiscriminator(nn.Module):
    """WGAN-GP discriminator.  Operates on GAP-pooled feature vectors."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // 2, channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        """x : [N, C, h, w]  →  scores : [N, 1]"""
        pooled = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        return self.net(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# LAWC — Learnable Attention-Weighted Correlation
# ─────────────────────────────────────────────────────────────────────────────

class LAWC(nn.Module):
    """Prototype-conditioned attention matching.

    For each prototype p ∈ P^(ell):
      1. Channel gate  a_c = σ(W_c · GAP(p))           [C]
      2. Spatial gate  a_s = σ(Conv1×1(F))              [H, W]
      3. Modulated map F'' = F ⊙ a_c ⊙ a_s             (broadcast to [H,W,C])
      4. Cosine sim    S_p(x,y) computed by sliding p as a conv filter over F'',
                       with local-patch normalisation → [H, W]
      5. Weighted sum  S_fused = Σ_p w_p · S_p          [H, W]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels     = channels
        self.W_c          = nn.Linear(channels, channels, bias=True)
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        # scalar weight per prototype — resized dynamically in forward
        self._proto_weights: nn.Parameter | None = None
        self._N_cached = 0

    def _get_proto_weights(self, N: int, device) -> torch.Tensor:
        if self._proto_weights is None or self._N_cached != N:
            self._proto_weights = nn.Parameter(
                torch.ones(N, device=device) / N
            )
            self._N_cached = N
        return self._proto_weights

    def _channel_gate(self, p: torch.Tensor) -> torch.Tensor:
        """p : [C, h, w]  →  a_c : [C]"""
        p_bar = p.mean(dim=[-2, -1])              # GAP → [C]
        return torch.sigmoid(self.W_c(p_bar))     # [C]

    def _spatial_gate(self, F: torch.Tensor) -> torch.Tensor:
        """F : [B, C, H, W]  →  a_s : [B, 1, H, W]"""
        return torch.sigmoid(self.conv_spatial(F))

    def _cosine_sim_map(
        self,
        F_mod: torch.Tensor,   # [B, C, H, W]
        p:     torch.Tensor,   # [C, h, w]
    ) -> torch.Tensor:         # [B, H, W]
        """Slide prototype p over F_mod as a convolutional filter.

        At each location (x,y), the cosine similarity between the local feature
        patch F''_{(x,y)} ∈ R^{h×w×C} and the prototype p ∈ R^{h×w×C} is:

            S_p(x,y) = <F''_{(x,y)}, p> / (||F''_{(x,y)}|| · ||p||)

        Implemented via F.conv2d with same-size padding so the output retains
        the full [H, W] spatial resolution.
        """
        C, h, w   = p.shape
        pad_h, pad_w = h // 2, w // 2

        # kernel shape: [1, C, h, w]
        kernel = p.unsqueeze(0)

        # numerator: local dot product at every location → [B, 1, H, W]
        numerator = F.conv2d(F_mod, kernel, padding=(pad_h, pad_w))

        # local patch norm: use squared conv with all-ones kernel → [B, 1, H, W]
        ones_kernel   = torch.ones_like(kernel)
        local_sq_sum  = F.conv2d(F_mod ** 2, ones_kernel,
                                  padding=(pad_h, pad_w)).clamp(min=1e-6)
        F_local_norm  = local_sq_sum.sqrt()

        p_norm = p.norm(p=2).clamp(min=1e-6)          # scalar

        S_p = (numerator / (F_local_norm * p_norm)).squeeze(1)  # [B, H, W]
        return S_p

    def forward(
        self,
        F:          torch.Tensor,   # [B, C, H, W]
        prototypes: torch.Tensor,   # [N, C, h, w]
    ) -> torch.Tensor:              # [B, H, W]
        B, C, H, W = F.shape
        N = prototypes.shape[0]

        # spatial gate — shared across all prototypes (depends only on image)
        a_s = self._spatial_gate(F)    # [B, 1, H, W]

        sim_maps = []
        for i in range(N):
            p   = prototypes[i]                        # [C, h, w]
            a_c = self._channel_gate(p).view(1, C, 1, 1)  # [1,C,1,1]
            F_mod = F * a_c * a_s                      # [B, C, H, W]
            sim_maps.append(self._cosine_sim_map(F_mod, p))  # [B, H, W]

        sim_stack = torch.stack(sim_maps, dim=1)       # [B, N, H, W]

        w = torch.softmax(
            self._get_proto_weights(N, F.device), dim=0
        ).view(1, N, 1, 1)

        return (sim_stack * w).sum(dim=1)              # [B, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Density Decoder + CGDM
# ─────────────────────────────────────────────────────────────────────────────

class DensityDecoder(nn.Module):
    """Progressive upsampling decoder with two parallel output heads.

    Input  : S_cat [B, 2, H/16, W/16]   (2-channel fused similarity map)
    Outputs: D_hat, D_raw, R  — all [B, 1, H, W]

    CGDM: D_hat = D_raw · R  (element-wise confidence gating)
    The reliability map R is learned implicitly from density supervision.
    """

    def __init__(self, input_channels: int = 2, base_channels: int = 64):
        super().__init__()

        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        C = base_channels
        self.block1 = _block(input_channels, C)
        self.up1    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block2 = _block(C,    C)
        self.up2    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block3 = _block(C,    C // 2)
        self.up3    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block4 = _block(C // 2, C // 4)
        self.up4    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block5 = _block(C // 4, C // 4)

        self.density_head     = nn.Conv2d(C // 4, 1, 1)
        self.reliability_head = nn.Conv2d(C // 4, 1, 1)

    def forward(self, S_cat: torch.Tensor):
        """
        S_cat : [B, 2, H3, W3]
        returns D_hat, D_raw, R — each [B, 1, H, W]
        """
        x = self.block1(S_cat)
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.block4(x)
        x = self.up4(x)
        x = self.block5(x)

        D_raw = F.relu(self.density_head(x))              # [B,1,H,W]  non-negative
        R     = torch.sigmoid(self.reliability_head(x))   # [B,1,H,W]  in (0,1)
        D_hat = D_raw * R                                  # CGDM
        return D_hat, D_raw, R


# ─────────────────────────────────────────────────────────────────────────────
# FICount — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FICount(nn.Module):
    """FICount end-to-end model.

    Args:
        M           : number of PVG variants per exemplar (default 2)
        noise_dim   : PVG noise vector dimension
        pool_size   : RoI-pool output spatial size
        decoder_base: base channel count for the decoder

    Forward:
        image       : [B, 3, H, W]
        boxes_list  : list of B tensors, each [K, 4] in image pixel coords [y1,x1,y2,x2]
        training    : bool — if True, returns pvg fake features for adv loss

    Returns:
        D_hat  : [B, 1, H, W]   final density (after CGDM)
        D_raw  : [B, 1, H, W]   raw density
        R      : [B, 1, H, W]   reliability map
        extras : dict  {'pvg_fakes3', 'pvg_fakes4', 'exemplars3', 'exemplars4'}
                        populated only when training=True
    """

    C3      = 1024
    C4      = 2048
    STRIDE3 = 16
    STRIDE4 = 32

    def __init__(
        self,
        M:            int = 2,
        noise_dim:    int = 256,
        pool_size:    int = 7,
        decoder_base: int = 64,
    ):
        super().__init__()
        self.M         = M
        self.pool_size = pool_size

        self.backbone = Resnet50FPN()

        self.pvg3  = PVG(self.C3, noise_dim=noise_dim, M=M)
        self.pvg4  = PVG(self.C4, noise_dim=noise_dim, M=M)

        self.lawc3 = LAWC(self.C3)
        self.lawc4 = LAWC(self.C4)

        self.decoder = DensityDecoder(input_channels=2, base_channels=decoder_base)

    # ── RoI extraction ──────────────────────────────────────────────────────

    def _roi_pool(
        self,
        feat_map: torch.Tensor,   # [1, C, Hf, Wf]
        boxes:    torch.Tensor,   # [K, 4]  [y1,x1,y2,x2] image coords
        stride:   int,
    ) -> torch.Tensor:            # [K, C, pool_size, pool_size]
        K      = boxes.shape[0]
        device = feat_map.device

        # scale to feature-map coordinates, convert to [x1,y1,x2,y2] for roi_align
        b_scaled  = boxes.float() / stride           # [K,4]
        rois_xyxy = b_scaled[:, [1, 0, 3, 2]]        # swap y↔x

        batch_idx = torch.zeros(K, 1, device=device)
        rois      = torch.cat([batch_idx, rois_xyxy], dim=1)   # [K, 5]

        return torchvision.ops.roi_align(
            feat_map,
            rois,
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=1.0,   # boxes already in feat-map coords
            aligned=True,
        )   # [K, C, pool_size, pool_size]

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, image, boxes_list, training=False):
        B      = image.shape[0]
        device = image.device

        feats = self.backbone(image)
        F3    = feats['map3']    # [B, 1024, H/16, W/16]
        F4    = feats['map4']    # [B, 2048, H/32, W/32]

        S3_list, S4_list = [], []
        all_fakes3, all_fakes4 = [], []
        all_real3,  all_real4  = [], []

        for b in range(B):
            boxes = boxes_list[b]   # [K, 4]

            # ── exemplar features via RoI pooling ────────────────────────
            ex3 = self._roi_pool(F3[b:b+1], boxes, self.STRIDE3)  # [K, C3, p, p]
            ex4 = self._roi_pool(F4[b:b+1], boxes, self.STRIDE4)  # [K, C4, p, p]

            # ── PVG: generate M variants ─────────────────────────────────
            var3 = self.pvg3(ex3)   # [K, M, C3, p, p]
            var4 = self.pvg4(ex4)   # [K, M, C4, p, p]

            if training:
                all_fakes3.append(var3.view(-1, self.C3, self.pool_size, self.pool_size))
                all_fakes4.append(var4.view(-1, self.C4, self.pool_size, self.pool_size))
                all_real3.append(ex3)
                all_real4.append(ex4)

            # ── build full prototype set: originals + variants ───────────
            gen3 = var3.view(-1, self.C3, self.pool_size, self.pool_size)
            gen4 = var4.view(-1, self.C4, self.pool_size, self.pool_size)
            proto3 = torch.cat([ex3, gen3], dim=0)   # [K*(1+M), C3, p, p]
            proto4 = torch.cat([ex4, gen4], dim=0)   # [K*(1+M), C4, p, p]

            # ── LAWC ─────────────────────────────────────────────────────
            S3_list.append(self.lawc3(F3[b:b+1], proto3))  # [1, H3, W3]
            S4_list.append(self.lawc4(F4[b:b+1], proto4))  # [1, H4, W4]

        # ── merge dual-scale streams ─────────────────────────────────────
        S3 = torch.cat(S3_list, dim=0).unsqueeze(1)   # [B, 1, H3, W3]
        S4 = torch.cat(S4_list, dim=0).unsqueeze(1)   # [B, 1, H4, W4]

        # upsample S4 to S3 resolution (factor-of-2 difference)
        S4_up   = F.interpolate(S4, size=S3.shape[-2:], mode='bilinear', align_corners=False)
        S_cat   = torch.cat([S3, S4_up], dim=1)       # [B, 2, H3, W3]

        # ── decoder + CGDM ───────────────────────────────────────────────
        D_hat, D_raw, R = self.decoder(S_cat)

        extras = {}
        if training and all_fakes3:
            extras['pvg_fakes3'] = torch.cat(all_fakes3, dim=0)
            extras['pvg_fakes4'] = torch.cat(all_fakes4, dim=0)
            extras['exemplars3'] = torch.cat(all_real3,  dim=0)
            extras['exemplars4'] = torch.cat(all_real4,  dim=0)

        return D_hat, D_raw, R, extras


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

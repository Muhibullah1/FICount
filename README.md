# FICount: Prototype-Guided Attention Matching for Few-Shot Insect Counting

# Overview
FICount is a few-shot insect counting framework designed for dense, heterogeneous sticky trap imagery. Given a query trap image and a small number of user-drawn exemplar crops (1–4 shots), FICount predicts a full-resolution density map whose spatial integral gives the total insect count.

# Architecture
Input Image ──► ResNet-50 (frozen)
                  ├── Layer 3 [B, 1024, H/16, W/16]
                  └── Layer 4 [B, 2048, H/32, W/32]

Exemplar Crops ──► RoI Pooling ──► PVG ──► [K×(1+M) prototypes per layer]

Image Features + Prototypes ──► LAWC ──► S_fused per layer
                                            ├── S_fused³  [B, 1, H/16, W/16]
                                            └── S_fused⁴  [B, 1, H/32, W/32]
                                                 ↓ upsample + concat
                                           S_cat [B, 2, H/16, W/16]
                                                 ↓ Decoder
                                           D_raw + R (reliability)
                                                 ↓ CGDM: D̂ = D_raw × R
                                           D̂    [B, 1, H, W]  → sum → count

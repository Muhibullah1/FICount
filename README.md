# FICount: Prototype-Guided Attention Matching for Few-Shot Insect Counting

# Overview
FICount is a few-shot insect counting framework designed for dense, heterogeneous sticky trap imagery. Given a query trap image and a small number of user-drawn exemplar crops (1–4 shots), FICount predicts a full-resolution density map whose spatial integral gives the total insect count.

# Installation
Requirements: Python 3.8+, PyTorch ≥ 1.12, CUDA 11.x recommended.

## Repository Structure

```
FICount/
├── model.py          # Full model: Resnet50FPN, PVG, PVGDiscriminator, LAWC, DensityDecoder
├── utils.py          # Transforms, losses (L_ex, WGAN-GP, L_id), visualisation
├── train.py          # End-to-end training script
├── test.py           # Evaluation script (MAE / RMSE)
├── demo.py           # Single-image inference with interactive box selection
├── data/             # Dataset directory (see Data Setup)
│    ├── Images/
│    ├── GTDensity_map/
│    ├── Annotation.json
│    └── TrainTestVal.json
└── logs/             # Checkpoints and training stats (created at runtime)
```

# Training
```
python train.py \
  --data_path  ./data/ \
  --output_dir ./logs/ \
  --epochs     1500 \
  --num_shots  4 \
  --num_variants 2 \
  --learning_rate 1e-4 \
  --lambda_ex   1.0 \
  --lambda_adv  0.1 \
  --lambda_id   1.0 \
  --gpu 0 
```
  # Evaluation
  ```python test.py \
  --data_path  ./data/ \
  --model_path ./logs/FICount_best.pth \
  --test_split test \
  --num_shots  4 \
  --gpu 0
```
To evaluate under the 1-shot setting:
```
python test.py --model_path ./logs/FICount_best.pth --num_shots 1 --test_split test
```
  # Demo
Interactive box selection
```
python demo.py \
  --input_image /path/to/trap_image.jpg \
  --model_path  ./logs/FICount_best.pth \
  --output_dir  ./output/ \
  --gpu 0
```

 # Acknowledgements
The image transform utilities and training loop structure are adapted from FamNet (CVPR 2021) by Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. The PVG design is inspired by the Adversarial Feature Hallucination Network (AFHN, CVPR 2020). The WGAN-GP training objective follows Gulrajani et al. (NeurIPS 2017). 

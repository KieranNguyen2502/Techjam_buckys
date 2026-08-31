# 5. Robust Detection of AI‑Generated Images Under Real‑World Transformations

Detecting AI-generated images with robustness to real-world post-processing (compression, blur, resizing, noise, color adjustment, cropping).

---

## 1. Project Overview

### Background
Generative AI tools make it easy to produce highly realistic synthetic images at scale, increasing risks around misinformation, impersonation, fraud, and platform trust. In practice, images are rarely encountered "clean" — they get compressed, cropped, resized, or lightly edited before redistribution — so a detector that only performs well on unmodified images is of limited real-world use.

### Problem
Build a prototype that distinguishes **AI-generated** images from **authentic** images, and that keeps working after common transformations: JPEG compression, Gaussian blur, downscaling/upscaling, Gaussian noise, color jitter, and center cropping — under a hackathon-scale compute budget and a **<2B parameter** model constraint.

### What We Have
- A training pipeline (`head_finetune.py`) that fine-tunes the classification head of **`google/efficientnet-b0`** (~5.3M parameters, well under the 2B limit) on the **[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)** dataset, which labels images as `real`, `synthetic`, or `tampered`.
- A SLURM job script (`job.sh`) that sets up the environment, trains the model with an automatic time-limit resubmission mechanism, and stops after a configurable max number of resubmission attempts.
- A single-image inference script (`predict.py`) that loads the fine-tuned model and prints the predicted class and per-class confidence scores for one image.
---

## 2. Approach

| Component | Choice | Why |
|---|---|---|
| Classes | `real`, `synthetic`, `tampered` (from SID_Set) | Matches the dataset's native labeling, giving finer-grained signal than a binary real/fake split |
| Base model | `google/efficientnet-b0` | Lightweight (~5.3M parameters), well below the 2B-parameter limit, and offers a strong accuracy–compute trade-off for the hackathon budget |
| Fine-tuning strategy | Freeze backbone, train classification head only | Reduces training time and GPU usage while still adapting the model to the AIGC detection task |
| Training infrastructure | SLURM batch job with auto-resubmission | Handles the cluster's hard time limit (`--time=115`, `SIGUSR1@120` signal) and allows training to resume from checkpoints without losing progress |
| Inference | HuggingFace `AutoModelForImageClassification` + `AutoImageProcessor` | Keeps preprocessing consistent between training and inference |

---

## 3. Repository Structure

```
.
├── .gitignore  
├── README.md
├── head_finetune.py  
├── job.sh
└── predict.py
```

---

## 4. Setup & Installation

**Requirements:** Python 3.10+, a CUDA-capable GPU (training was run with CUDA 12.6), and SLURM if using `job.sh` as-is.

```bash
# 1. Clone the repository
git clone https://github.com/KieranNguyen2502/Techjam_buckys.git
cd Techjam_buckys

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets evaluate scikit-learn numpy Pillow
```

---

## 5. Steps to Reproduce

### Train the model
- **On a SLURM cluster:**
  ```bash
  sbatch job.sh
  ```

- **Locally / without SLURM:**
  ```bash
  python head_finetune.py
  ```
---

## 6. Results

> Placeholder — to be filled in once the batch-inference and robustness-evaluation scripts are complete.

| Transform | Parameters | Accuracy |
|---|---|---|
| Clean (no transform) | | TBD |
| JPEG | $quality = 90 / 70 / 50 / 30$ | TBD |
| Gaussian blur | $σ = 0.5 / 1.0 / 2.0$ | TBD |
| Resize | scale $0.5\times$ / $0.25\times$ → upscale | TBD |
| Gaussian noise | $σ = 0.02 / 0.05 / 0.10$ | TBD |
| Color jitter | $±20\%$ brightness/contrast/sat. | TBD |
| Center | crop $80\%$ | TBD |

---
## 7. License & Acknowledgements

- Base model: [`google/efficientnet-b0`](https://huggingface.co/google/efficientnet-b0)
- Dataset: [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set)

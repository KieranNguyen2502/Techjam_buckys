#!/bin/sh
#SBATCH --job-name=efficientnet-train
#SBATCH --time=115
#SBATCH --gpus=1
#SBATCH --signal=SIGUSR1@120
#SBATCH --output=logs/slurm-%j.out

# Uninstall old torch versions
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 12.6 support from official index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies from PyPI
pip install transformers datasets evaluate scikit-learn numpy

ATTEMPT_FILE=".train_attempt_count"
ATTEMPT=$(cat "$ATTEMPT_FILE" 2>/dev/null || echo 0)
MAX_ATTEMPTS=20

if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    echo "Hit max resubmission attempts ($MAX_ATTEMPTS) — stopping. Investigate before resubmitting manually."
    exit 1
fi
echo $((ATTEMPT + 1)) > "$ATTEMPT_FILE"

srun python head_finetune.py
EXIT_CODE=$?

if [ -d "efficientnet-b0-finetuned/final" ]; then
    echo "Training complete."
    rm -f "$ATTEMPT_FILE"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "script.py exited with an error (code $EXIT_CODE) — NOT resubmitting automatically. Check logs/slurm-${SLURM_JOB_ID}.out."
else
    echo "Time limit reached before training finished — resubmitting."
    sbatch "$(cd "$(dirname "$0")" && pwd)/job.sh" --dependency=afterany:$SLURM_JOB_ID
fi
import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch
from torchvision.transforms import (
    Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop,
    Resize, CenterCrop, ToTensor,
)

REPO_ID = "saberzl/SID_Set"

# Load dataset
# NOTE: this file uses the same dataset and transforms as the original script.
dataset = load_dataset(REPO_ID)

id2label = {0: "real", 1: "synthetic", 2: "tampered"}
label2id = {v: k for k, v in id2label.items()}
labels = list(id2label.values())

checkpoint = "google/efficientnet-b2"
processor = AutoImageProcessor.from_pretrained(checkpoint)

size = (
    processor.size["shortest_edge"]
    if "shortest_edge" in processor.size
    else (processor.size["height"], processor.size["width"])
)
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

train_transforms = Compose(
    [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize]
)
val_transforms = Compose(
    [Resize(size), CenterCrop(size), ToTensor(), normalize]
)


def apply_train_transforms(examples):
    examples["pixel_values"] = [
        train_transforms(img.convert("RGB")) for img in examples["image"]
    ]
    return examples


def apply_val_transforms(examples):
    examples["pixel_values"] = [
        val_transforms(img.convert("RGB")) for img in examples["image"]
    ]
    return examples


train_key = "train"
eval_key = "validation" if "validation" in dataset else "test"

dataset[train_key].set_transform(apply_train_transforms)
dataset[eval_key].set_transform(apply_val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Load pretrained EfficientNet model
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Freeze the backbone and keep only the classification head trainable
for name, param in model.named_parameters():
    if not name.startswith("classifier"):
        param.requires_grad = False

# Optional: print number of trainable parameters for verification
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}")

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


OUTPUT_DIR = "efficientnet-b2-head-finetuned"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    warmup_steps=200,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset[train_key],
    eval_dataset=dataset[eval_key],
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    processor.save_pretrained(f"{OUTPUT_DIR}/final")

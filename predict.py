import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def find_model_dir(model_dir: str | None) -> Path:
    if model_dir:
        p = Path(model_dir).expanduser().resolve()
    else:
        p = Path("final").expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"Model folder not found: {p}. Make sure the copied model files are inside the final folder."
        )
    return p


def predict(model_dir: str | None, image_path: str) -> None:
    model_path = find_model_dir(model_dir)
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = AutoModelForImageClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    id2label = model.config.id2label
    label = id2label.get(pred_idx, str(pred_idx))

    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.4f}")
    print("Class probabilities:")
    for idx, p in enumerate(probs):
        cls_name = id2label.get(int(idx), str(int(idx)))
        print(f"  {cls_name}: {float(p):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned EfficientNet model")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model_dir",
        default="final",
        help="Directory with the fine-tuned model. Default: ./final",
    )
    args = parser.parse_args()

    predict(args.model_dir, args.image)

#testmodel.py

import argparse
import json
from pathlib import Path

import requests

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def iter_images(root: Path):
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        true_label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, true_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)                 
    ap.add_argument("--root", required=True)                
    ap.add_argument("--field", default="file")              
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.root)
    items = list(iter_images(root))
    if not items:
        raise SystemExit("No images found. Check --root folder structure.")

    total = len(items)
    correct = 0
    sent = 0

    for img_path, true_label in items:
        with img_path.open("rb") as f:
            files = {args.field: (img_path.name, f, "application/octet-stream")}
            r = requests.post(args.url, files=files, timeout=args.timeout)

        if r.status_code != 200:
            sent += 1
            print(f"WRONG  {img_path.name}  true={true_label}  status={r.status_code}")
            continue

        data = json.loads(r.text)
        pred = str(data.get(args.label_key, data.get("label", "")))

        sent += 1
        if pred == true_label:
            correct += 1
            print(f"RIGHT  {img_path.name}  true={true_label}  pred={pred}")
        else:
            print(f"WRONG  {img_path.name}  true={true_label}  pred={pred}")

    accuracy = correct / total
    print("\n=== FINAL ===")
    print(f"Total images: {total}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {accuracy:.6f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
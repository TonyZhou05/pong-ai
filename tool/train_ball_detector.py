"""Fine-tune a YOLO11n ball detector on the OpenTTGames ball labels.

The stock COCO `sports ball` class is the corpus pipeline's weakest link: a
40 mm ball motion-blurs to a few faint pixels, so detector recall in the
unlabeled stretches (segment extensions, live camera) is poor. OpenTTGames
ships per-frame ball centres for six videos — exactly the supervision needed
to specialise the detector for this camera geometry.

Builds a single-class YOLO dataset (train = test_1/2/3/5/6, val = test_7 for
a held-out video) and fine-tunes from yolo11n.pt. The resulting
`runs/detect/<name>/weights/best.pt` slots into the extractor's `det_model`
and, exported to .tflite/.mlpackage, into the app's `pingPongDetectProfile`.

Usage (from the directory holding openttgames/):
  python3 train_ball_detector.py [--epochs 30] [--imgsz 960]
"""

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path.cwd() / "openttgames"
OUT = Path.cwd() / "ball_dataset"
BALL_BOX_PX = 22  # slightly generous box around the labeled centre
FRAME_STRIDE = 2  # use every 2nd labeled frame (labels are near-consecutive)

VIDEOS = {
    "train": ["test_1", "test_2", "test_3", "test_5", "test_6"],
    "val": ["test_7"],
}


def build_split(split, names):
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in names:
        ball = {
            int(k): v
            for k, v in json.loads(
                (ROOT / f"{name}_markup/ball_markup.json").read_text()
            ).items()
        }
        frames = sorted(
            f
            for f, p in ball.items()
            if 5 < p["x"] < 1915 and 5 < p["y"] < 1075  # drop junk coords
        )[::FRAME_STRIDE]
        cap = cv2.VideoCapture(str(ROOT / f"{name}.mp4"))
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            stem = f"{name}_{f:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            p = ball[f]
            (lbl_dir / f"{stem}.txt").write_text(
                f"0 {p['x']/W:.6f} {p['y']/H:.6f} "
                f"{BALL_BOX_PX/W:.6f} {BALL_BOX_PX/H:.6f}\n"
            )
            total += 1
        cap.release()
        print(f"  {name}: done (cumulative {total} images in {split})")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    if not (OUT / "data.yaml").exists():
        print("Building dataset…")
        n_train = build_split("train", VIDEOS["train"])
        n_val = build_split("val", VIDEOS["val"])
        (OUT / "data.yaml").write_text(
            f"path: {OUT}\ntrain: images/train\nval: images/val\n"
            "names:\n  0: ball\n"
        )
        print(f"dataset: {n_train} train / {n_val} val images")
    else:
        print("dataset already built")

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(OUT / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=8,
        device="mps",
        name="pingpong_ball",
        patience=10,
        plots=False,
    )
    print("training complete — best weights at "
          "runs/detect/pingpong_ball/weights/best.pt")


if __name__ == "__main__":
    main()

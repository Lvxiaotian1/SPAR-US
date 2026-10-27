"""Render the frames of a fitted SPAR-US checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from model import load_checkpoint, load_images, render


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render the frames described by a SPAR-US checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fitted checkpoint")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--data", type=Path, default=None, help="optional directory holding images/; enables image metrics")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    field, probe = load_checkpoint(args.checkpoint, args.device)

    images = None
    if args.data is not None and (args.data / "images").is_dir():
        images = load_images(args.data / "images", args.device)
        if tuple(images.shape[1:]) != (probe.height, probe.width):
            raise ValueError("Image dimensions do not match the checkpoint.")

    args.output.mkdir(parents=True, exist_ok=True)
    count = int(probe.phi.shape[0])
    predictions, metrics = [], []
    with torch.inference_mode():
        for index in range(count):
            predicted = render(field, probe, index, config)
            array = predicted.cpu().numpy()
            predictions.append(array)
            png = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
            if not cv2.imwrite(str(args.output / f"frame_{index:02d}.png"), png):
                raise OSError("Could not save a rendered image.")
            if images is not None:
                mse = float(torch.mean((predicted - images[index]) ** 2).item())
                psnr = float("inf") if mse == 0 else float(-10 * np.log10(mse))
                metrics.append({"frame": index, "mse": mse, "psnr_db": psnr})
                print(f"Frame {index:02d}: MSE={mse:.8f}, PSNR={psnr:.3f} dB", flush=True)
            else:
                print(f"Frame {index:02d}: rendered", flush=True)
        poses = probe(slice(None)).cpu().numpy()

    np.save(args.output / "rendered.npy", np.stack(predictions))
    np.save(args.output / "poses.npy", poses)
    if metrics:
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "mse", "psnr_db"])
            writer.writeheader()
            writer.writerows(metrics)
    print(f"Saved {count} frames and poses to {args.output}.")


if __name__ == "__main__":
    main()

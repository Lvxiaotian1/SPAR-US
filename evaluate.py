"""Score rendered frames against reference poses and the input images."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity


def homogeneous(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] not in [(3, 4), (4, 4)]:
        raise ValueError("Pose arrays must have shape (N, 3, 4) or (N, 4, 4).")
    if len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("Pose arrays must contain at least two finite poses.")
    if poses.shape[1] == 4 and not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Homogeneous pose matrices must end in [0, 0, 0, 1].")
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3] = poses[:, :3]
    for pose in result:
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0) or np.linalg.det(rotation) <= 0:
            raise ValueError("Pose rotations must be right-handed orthonormal matrices.")
        u, _, vt = np.linalg.svd(rotation)
        corrected = u @ vt
        if np.linalg.det(corrected) < 0:
            u[:, -1] *= -1
            corrected = u @ vt
        pose[:3, :3] = corrected
    return result


def pose_errors(predicted, reference):
    predicted, reference = homogeneous(predicted), homogeneous(reference)
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference pose counts must match.")
    anchor = int(np.argmin(reference[:, 2, 3]))
    predicted = np.linalg.inv(predicted[anchor]) @ predicted
    reference = np.linalg.inv(reference[anchor]) @ reference
    translation = np.linalg.norm(predicted[:, :3, 3] - reference[:, :3, 3], axis=1)
    relative = predicted[:, :3, :3] @ reference[:, :3, :3].transpose(0, 2, 1)
    rotation = np.degrees(Rotation.from_matrix(relative).magnitude())
    included = np.arange(len(predicted)) != anchor
    return translation, rotation, included, anchor


def image_ssim(reference_path, predicted_path):
    if not reference_path.is_file() or not predicted_path.is_file():
        raise ValueError("Missing image pair: images/<id>.png and results/frame_<id:02d>.png are required.")
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    predicted = cv2.imread(str(predicted_path), cv2.IMREAD_GRAYSCALE)
    if reference is None or predicted is None or reference.shape != predicted.shape:
        raise ValueError("Image pairs must be readable and have identical dimensions.")
    if min(reference.shape) < 7:
        raise ValueError("SSIM requires image dimensions of at least 7 pixels.")
    reference = reference.astype(np.float64) / 255.0
    predicted = predicted.astype(np.float64) / 255.0
    return float(structural_similarity(reference, predicted, data_range=1.0, win_size=7, gaussian_weights=False, use_sample_covariance=True))


def evaluate(data, results):
    if not (data / "poses.npy").is_file():
        raise ValueError(f"Missing reference poses: {data / 'poses.npy'}.")
    if not (results / "poses.npy").is_file():
        raise ValueError(f"Missing estimated poses: {results / 'poses.npy'}. Run demo.py first.")
    reference = np.load(data / "poses.npy", allow_pickle=False)
    predicted = np.load(results / "poses.npy", allow_pickle=False)
    translation, rotation, included, anchor = pose_errors(predicted, reference)
    count = len(translation)
    expected = {f"{index}.png" for index in range(count)}
    if {path.name for path in (data / "images").glob("*.png")} != expected:
        raise ValueError(f"Frames in {data / 'images'} must be 0.png through {count - 1}.png.")
    rows = []
    for index in range(count):
        ssim = image_ssim(data / "images" / f"{index}.png", results / f"frame_{index:02d}.png")
        rows.append({
            "frame": index,
            "included_in_pose_mean": bool(included[index]),
            "translation_mm": float(translation[index]),
            "rotation_deg": float(rotation[index]),
            "ssim": ssim,
        })
    summary = {
        "frames": count,
        "pose_frames": int(included.sum()),
        "alignment_frame": anchor,
        "translation_mm": float(translation[included].mean()),
        "rotation_deg": float(rotation[included].mean()),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
    }
    return rows, summary


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate translation, rotation and SSIM.")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--results", type=Path, default=root / "results")
    args = parser.parse_args()
    try:
        rows, summary = evaluate(args.data, args.results)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    with (args.results / "evaluation_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.results / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Translation: {summary['translation_mm']:.6f} mm")
    print(f"Rotation: {summary['rotation_deg']:.6f} degrees")
    print(f"SSIM: {summary['ssim']:.6f}")


if __name__ == "__main__":
    main()

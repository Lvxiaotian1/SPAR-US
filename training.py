"""Image-pose initialisation and progressive pose-guided reconstruction.

A single frame set is described by its frames, a three-group assignment and a
pairwise similarity matrix. The initial spatial layout follows the scan-range
prior supplied by the caller; no reference pose is used during fitting.
"""
import random
from pathlib import Path

import numpy as np
import torch

from geometry import matrix_to_axis_angle
from model import Field, UltrasoundProbe, load_images, render


def tilt_rotation(degrees):
    """Rotation about the lateral axis by ``degrees``, in the frame's own axes."""
    theta = torch.tensor(float(degrees) * np.pi / 180.0, dtype=torch.float32)
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        (
            torch.stack((torch.ones_like(cos), torch.zeros_like(cos), torch.zeros_like(cos))),
            torch.stack((torch.zeros_like(cos), cos, -sin)),
            torch.stack((torch.zeros_like(cos), sin, cos)),
        )
    )


def initialize_probe(images, classes, similarity, scan_range_mm, tilt_deg, config):
    """Place the frames on the scan axis from similarity and the scan-range prior.

    Groups are ordered along the sweep direction, their spans are proportional to
    their frame counts, and the within-group order follows the similarity to each
    group anchor. The three anchor frames start one per group.
    """
    labels, counts = np.unique(classes, return_counts=True)
    if len(labels) != 3 or similarity.shape != (len(classes), len(classes)):
        raise ValueError("Expected three frame groups and a square similarity matrix.")
    anchors = [int(np.flatnonzero(classes == label)[0]) for label in labels]
    fractions = np.empty(len(classes), dtype=np.float64)
    left = 0.0
    for group, (label, count) in enumerate(zip(labels, counts)):
        width = count / len(classes)
        anchor = anchors[group]
        others = [int(index) for index in np.flatnonzero(classes == label) if index != anchor]
        if group == 0:
            order = [anchor] + sorted(others, key=lambda i: (-float(similarity[i, anchor]), i))
            points = left + np.arange(count) * width / count
        elif group == 2:
            order = sorted(others, key=lambda i: (float(similarity[i, anchor]), i)) + [anchor]
            points = left + (np.arange(count) + 1) * width / count
        else:
            before = [i for i in others if similarity[i, anchors[0]] >= similarity[i, anchors[2]]]
            after = [i for i in others if similarity[i, anchors[0]] < similarity[i, anchors[2]]]
            order = sorted(before, key=lambda i: (float(similarity[i, anchor]), i)) + [anchor]
            order += sorted(after, key=lambda i: (-float(similarity[i, anchor]), i))
            points = left + (np.arange(count) + 0.5) * width / count
        fractions[order] = points
        left += width

    rotation = tilt_rotation(tilt_deg)
    normal = torch.tensor([0.0, 0.0, 1.0]) @ rotation.T
    translation = torch.as_tensor(fractions[:, None], dtype=torch.float32) * float(scan_range_mm) * normal[None]
    phi = matrix_to_axis_angle(rotation).repeat(len(classes), 1)
    return UltrasoundProbe(phi, translation, images.shape[1], images.shape[2], config["sw"], config["sh"])


def stages(count, config, anchors):
    """Yield the optimisation stages: joint initialisation, local, partial, global."""
    yield "init", list(anchors), config["steps_init"]
    incorporated = list(anchors)
    for index in range(count):
        if index in anchors:
            continue
        yield "local", [index], config["steps_stage"]
        incorporated.append(index)
        yield "part", index, config["steps_stage"]
        if len(incorporated) % config["global_every"] == 0 or len(incorporated) == count:
            yield "global", list(incorporated), config["steps_stage"]


def local_subset(probe, incorporated, target, limit):
    """The target frame plus its nearest already-incorporated frames.

    Neighbours are ranked by the Euclidean distance between the current
    translation estimates, and at most ``limit - 1`` of them are kept.
    """
    others = [index for index in incorporated if index != target]
    if not others or limit <= 1:
        return [target]
    positions = probe.t.detach()
    distances = torch.linalg.vector_norm(positions[others] - positions[target], dim=1)
    nearest = torch.argsort(distances, stable=True)[: limit - 1].tolist()
    return [target] + [others[position] for position in nearest]


def optimize_stage(field, probe, images, stage, indices, steps, config):
    """Run one stage and return the mean pass loss."""
    field_optimizer = torch.optim.Adam(field.parameters(), lr=config["lr_field"], betas=(0.9, 0.999))
    pose_optimizer = torch.optim.Adam([probe.phi, probe.t], lr=config["lr_pose"], betas=(0.9, 0.999))
    field_scheduler = torch.optim.lr_scheduler.StepLR(field_optimizer, config["field_step"], config["field_decay"])
    pose_scheduler = torch.optim.lr_scheduler.StepLR(pose_optimizer, config["pose_step"], config["pose_decay"])
    running = 0.0
    for _ in range(steps):
        order = list(indices)
        if stage != "local":
            np.random.shuffle(order)
        for index in order:
            predicted = render(field, probe, index, config)
            loss = torch.nn.functional.mse_loss(predicted, images[index])
            field_optimizer.zero_grad(set_to_none=True)
            pose_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if stage != "local":
                field_optimizer.step()
                field_scheduler.step()
            pose_optimizer.step()
            pose_scheduler.step()
        running += float(loss.detach())
    return running / max(steps, 1)


def train_case(config, device, data_dir, scan_range_mm, tilt_deg=0.0, progress=None):
    """Fit one frame set. Returns the field, the estimated geometry and update count."""
    data_dir = Path(data_dir)
    images = load_images(data_dir / "images", device)
    classes = np.load(data_dir / "class.npy", allow_pickle=False).reshape(-1)
    similarity = np.load(data_dir / "similarity.npy", allow_pickle=False)
    if len(classes) != len(images):
        raise ValueError("class.npy must contain one label per frame.")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    field = Field().to(images.device)
    probe = initialize_probe(images, classes, similarity, scan_range_mm, tilt_deg, config).to(images.device)
    anchors = [int(np.flatnonzero(classes == label)[0]) for label in np.unique(classes)]
    incorporated = list(anchors)
    updates = 0
    for stage, frames, steps in stages(len(images), config, anchors):
        if stage == "part":
            indices = local_subset(probe, incorporated, frames, config["partial_views"])
        else:
            indices = list(frames)
            if stage == "local":
                incorporated.append(indices[0])
        loss = optimize_stage(field, probe, images, stage, indices, steps, config)
        updates += steps * len(indices)
        if progress is not None:
            progress(stage, len(indices), steps, loss, updates)
    return field, probe, updates

"""Phase 3 -- circular mean averaging across clips to produce canonical G1 motion."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import circmean, circstd
from scipy.interpolate import interp1d
import yaml

JOINT_KEYS: list[str] = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw",
    "kLeftKnee",
    "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw",
    "kRightKnee",
    "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw", "kWaistRoll", "kWaistPitch",
]

CONFIG_PATH = "pipeline_config.yaml"


def load_keyframe_files(dance_name: str, include_flagged: bool = False) -> list[dict]:
    """Load all valid _keyframes.json files for the dance, return list of G1 JSON dicts."""
    base_dir = Path("dances") / dance_name / "processed"
    results: list[dict] = []

    for keyframes_path in sorted(base_dir.glob("*/*_keyframes.json")):
        clip_name = keyframes_path.stem.replace("_keyframes", "")
        meta_path = keyframes_path.with_name(f"{clip_name}_meta.json")

        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            if meta.get("flagged", False) and not include_flagged:
                rate = meta.get("detection_rate", 0.0)
                print(f"[AVG] Skipping flagged clip: {clip_name} (detection_rate: {rate:.1%})")
                continue

        with keyframes_path.open() as f:
            data = json.load(f)

        if "MotionSwitcherClient" not in data:
            print(f"[AVG] WARNING: Skipping {clip_name}: missing MotionSwitcherClient key")
            continue

        frames = data["MotionSwitcherClient"].get("frames", [])
        if len(frames) < 1:
            print(f"[AVG] WARNING: Skipping {clip_name}: no frames")
            continue

        data["_clip_name"] = clip_name
        results.append(data)
        print(f"[AVG] Loaded: {clip_name} ({len(frames)} frames)")

    print(f"[AVG] Total clips loaded: {len(results)}")
    return results


def align_clip_lengths(clips: list[dict]) -> list[dict]:
    """Interpolate or pad all clips to match the longest clip's frame count."""
    target_frames = max(
        len(clip["MotionSwitcherClient"]["frames"]) for clip in clips
    )
    print(f"[AVG] Aligning {len(clips)} clips to {target_frames} frames")

    results: list[dict] = []
    for clip in clips:
        msc = clip["MotionSwitcherClient"]
        frames = msc["frames"]
        clip_name = clip.get("_clip_name", msc.get("name", "unknown"))

        if len(frames) == target_frames:
            results.append(clip)
            continue

        fps = msc["fps"]
        original = len(frames)
        resampled_joints: dict[str, np.ndarray] = {}
        x_target = np.linspace(0, 1, target_frames)

        for joint in JOINT_KEYS:
            series = np.array([f["joint_positions"][joint] for f in frames])
            interp_fn = interp1d(
                np.linspace(0, 1, len(series)),
                series,
                kind="linear",
            )
            resampled_joints[joint] = interp_fn(x_target)

        new_frames = [
            {
                "time": round(i / fps, 6),
                "joint_positions": {
                    joint: float(resampled_joints[joint][i])
                    for joint in JOINT_KEYS
                },
            }
            for i in range(target_frames)
        ]

        aligned_clip = {
            "MotionSwitcherClient": {
                "name": msc["name"],
                "fps": fps,
                "frames": new_frames,
            },
            "_clip_name": clip_name,
        }
        print(f"[AVG] Interpolated {clip_name} from {original} -> {target_frames} frames")
        results.append(aligned_clip)

    return results


def circular_mean_frames(clips: list[dict]) -> list[dict]:
    """Per-frame per-joint circular mean across all clips. Returns averaged frame list."""
    n_frames = len(clips[0]["MotionSwitcherClient"]["frames"])
    fps = clips[0]["MotionSwitcherClient"]["fps"]

    angles_matrix: dict[str, np.ndarray] = {}
    for joint in JOINT_KEYS:
        angles_matrix[joint] = np.array([
            [clip["MotionSwitcherClient"]["frames"][i]["joint_positions"][joint]
             for i in range(n_frames)]
            for clip in clips
        ])

    averaged_frames: list[dict] = []
    for i in range(n_frames):
        joint_positions = {
            joint: float(circmean(angles_matrix[joint][:, i], low=-np.pi, high=np.pi))
            for joint in JOINT_KEYS
        }
        averaged_frames.append({
            "time": round(i / fps, 6),
            "joint_positions": joint_positions,
        })

    print(f"[AVG] Circular mean computed across {len(clips)} clips x {n_frames} frames")
    return averaged_frames


def compute_confidence(clips: list[dict], averaged: list[dict]) -> dict:
    """Compute per-joint per-frame std deviation as confidence map."""
    n_frames = len(averaged)

    angles_matrix: dict[str, np.ndarray] = {}
    for joint in JOINT_KEYS:
        angles_matrix[joint] = np.array([
            [clip["MotionSwitcherClient"]["frames"][i]["joint_positions"][joint]
             for i in range(n_frames)]
            for clip in clips
        ])

    joints: dict[str, dict] = {}
    for joint in JOINT_KEYS:
        std_series = np.array([
            circstd(angles_matrix[joint][:, i], low=-np.pi, high=np.pi)
            for i in range(n_frames)
        ])
        joints[joint] = {
            "mean_std": float(np.mean(std_series)),
            "max_std": float(np.max(std_series)),
            "per_frame_std": std_series.tolist(),
        }

    print("[AVG] Confidence computed. Highest variance joints:")
    top_joints = sorted(joints.items(), key=lambda x: x[1]["mean_std"], reverse=True)[:5]
    for joint, stats in top_joints:
        print(f"  {joint}: mean_std={stats['mean_std']:.4f}")

    return {
        "n_clips": len(clips),
        "n_frames": n_frames,
        "joints": joints,
    }


def main() -> None:
    """CLI entry point. Parses --dance, --min-clips, --include-flagged args."""
    parser = argparse.ArgumentParser(description="Average pose keyframes across clips.")
    parser.add_argument("--dance", required=True, help="Dance folder name")
    parser.add_argument("--min-clips", type=int, default=3, help="Minimum clips required")
    parser.add_argument("--include-flagged", action="store_true", help="Include low-detection clips")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    min_clips = args.min_clips or config.get("min_clips_for_average", 3)

    clips = load_keyframe_files(args.dance, include_flagged=args.include_flagged)

    if len(clips) < min_clips:
        print(f"[AVG] ERROR: Only {len(clips)} valid clips found, need at least {min_clips}.")
        print("[AVG] Use --include-flagged to include low-detection clips, or --min-clips to lower the threshold.")
        sys.exit(1)

    aligned = align_clip_lengths(clips)
    averaged_frames = circular_mean_frames(aligned)

    dance_key = args.dance.replace(" ", "_").lower()
    averaged_json = {
        "MotionSwitcherClient": {
            "name": dance_key,
            "fps": aligned[0]["MotionSwitcherClient"]["fps"],
            "frames": averaged_frames,
        }
    }

    confidence = compute_confidence(aligned, averaged_frames)

    out_dir = Path("dances") / args.dance / "averaged"
    out_dir.mkdir(parents=True, exist_ok=True)

    averaged_path = out_dir / f"{dance_key}_averaged.json"
    confidence_path = out_dir / f"{dance_key}_confidence.json"

    with averaged_path.open("w") as f:
        json.dump(averaged_json, f, indent=2)
    print(f"[AVG] Averaged motion written: {averaged_path}")

    with confidence_path.open("w") as f:
        json.dump(confidence, f, indent=2)
    print(f"[AVG] Confidence map written: {confidence_path}")

    print(f"[AVG] Done. {len(aligned)} clips x {len(averaged_frames)} frames -> {averaged_path}")


if __name__ == "__main__":
    main()

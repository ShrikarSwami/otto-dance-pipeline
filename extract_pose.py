"""Phase 1 -- single video pose extraction and G1 joint angle conversion."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml
from scipy.signal import savgol_filter
from tqdm import tqdm

JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "kLeftShoulderPitch":  (-3.14, 2.53),
    "kLeftShoulderRoll":   (-0.09, 2.53),
    "kLeftShoulderYaw":    (-1.57, 1.57),
    "kLeftElbow":          (-1.57, 0.0),
    "kLeftWristRoll":      (-1.57, 1.57),
    "kLeftWristPitch":     (-0.87, 0.52),
    "kLeftWristYaw":       (-1.57, 1.57),
    "kRightShoulderPitch": (-3.14, 2.53),
    "kRightShoulderRoll":  (-2.53, 0.09),
    "kRightShoulderYaw":   (-1.57, 1.57),
    "kRightElbow":         (0.0, 1.57),
    "kRightWristRoll":     (-1.57, 1.57),
    "kRightWristPitch":    (-0.87, 0.52),
    "kRightWristYaw":      (-1.57, 1.57),
    "kLeftHipPitch":       (-1.75, 2.09),
    "kLeftHipRoll":        (-0.09, 0.79),
    "kLeftHipYaw":         (-0.79, 0.79),
    "kLeftKnee":           (-0.09, 2.09),
    "kLeftAnklePitch":     (-0.87, 0.52),
    "kLeftAnkleRoll":      (-0.44, 0.44),
    "kRightHipPitch":      (-1.75, 2.09),
    "kRightHipRoll":       (-0.79, 0.09),
    "kRightHipYaw":        (-0.79, 0.79),
    "kRightKnee":          (-0.09, 2.09),
    "kRightAnklePitch":    (-0.87, 0.52),
    "kRightAnkleRoll":     (-0.44, 0.44),
    "kWaistYaw":           (-0.52, 0.52),
    "kWaistRoll":          (-0.2, 0.2),
    "kWaistPitch":         (-0.52, 0.52),
}

JOINT_KEYS: list[str] = list(JOINT_LIMITS.keys())


def load_config(config_path: str = "pipeline_config.yaml") -> dict:
    """Load pipeline_config.yaml and return as dict."""
    path = Path(config_path)
    if not path.exists():
        print(f"[POSE] ERROR: pipeline_config.yaml not found at {config_path}")
        raise FileNotFoundError(config_path)
    with path.open() as f:
        return yaml.safe_load(f)


def setup_output_dirs(dance_name: str, clip_name: str) -> dict[str, Path]:
    """Create processed/{clip_name}/ directory and return dict of output paths."""
    base = Path("dances") / dance_name
    processed_dir = base / "processed" / clip_name
    averaged_dir = base / "averaged"
    processed_dir.mkdir(parents=True, exist_ok=True)
    averaged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[POSE] Output dir: {processed_dir}")
    return {
        "raw_dir": base / "raw",
        "processed_dir": processed_dir,
        "averaged_dir": averaged_dir,
        "original_video": base / "processed" / clip_name / f"{clip_name}_original.mp4",
        "pose_video": base / "processed" / clip_name / f"{clip_name}_pose.mp4",
        "keyframes_json": base / "processed" / clip_name / f"{clip_name}_keyframes.json",
        "meta_json": base / "processed" / clip_name / f"{clip_name}_meta.json",
    }


def extract_keypoints_vitpose(video_path: Path, config: dict) -> list[dict]:
    """Run ViTPose on each frame, return list of per-frame keypoint dicts."""
    pass


def extract_keypoints_mediapipe(video_path: Path, config: dict) -> list[dict]:
    """Run MediaPipe Pose Landmarker on each frame, return list of per-frame keypoint dicts."""
    import urllib.request
    import tempfile

    model_tier = config.get("mediapipe_model", "heavy").lower()
    model_specs = {
        "lite": (
            "pose_landmarker_lite.task",
            "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        ),
        "full": (
            "pose_landmarker_full.task",
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        ),
        "heavy": (
            "pose_landmarker_heavy.task",
            "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        ),
    }
    if model_tier not in model_specs:
        print(f"[POSE] WARNING: Unknown mediapipe_model '{model_tier}', using heavy")
        model_tier = "heavy"

    model_filename, model_suffix = model_specs[model_tier]
    model_path = Path(tempfile.gettempdir()) / model_filename
    if not model_path.exists():
        model_url = f"https://storage.googleapis.com/mediapipe-models/{model_suffix}"
        print(f"[POSE] Downloading MediaPipe {model_tier} model to {model_path} ...")
        urllib.request.urlretrieve(model_url, model_path)
        print("[POSE] Model download complete.")

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode

    delegate_name = config.get("mediapipe_delegate", "gpu").lower()
    delegate = mp_python.BaseOptions.Delegate.CPU
    if delegate_name == "gpu":
        delegate = mp_python.BaseOptions.Delegate.GPU
    print(f"[POSE] MediaPipe model: {model_tier}, delegate: {delegate_name}")

    base_options = mp_python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=delegate,
    )
    options = PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=config["confidence_threshold"],
        min_pose_presence_confidence=config["confidence_threshold"],
        min_tracking_confidence=config["confidence_threshold"],
        running_mode=RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"[POSE] ERROR: Cannot open video {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        print("[POSE] WARNING: Could not read source FPS, assuming 30")
        source_fps = 30.0

    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = config["target_fps"]
    raw_frames: list[dict] = []

    try:
        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            for frame_idx in tqdm(
                range(total_raw_frames),
                desc="[POSE] Extracting keypoints (MediaPipe)",
                unit="frame",
            ):
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Timestamp in milliseconds for VIDEO mode
                timestamp_ms = int(frame_idx * 1000 / source_fps)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                    raw_frames.append({
                        "frame_idx": frame_idx,
                        "keypoints": [[0.0, 0.0, 0.0, 0.0]] * 33,
                        "scores": [0.0] * 33,
                        "detected": False,
                        "person_bbox": [0.0, 0.0, 0.0, 0.0],
                    })
                    continue

                landmarks = result.pose_landmarks[0]  # num_poses=1
                keypoints = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
                scores = [lm.visibility for lm in landmarks]

                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                person_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

                raw_frames.append({
                    "frame_idx": frame_idx,
                    "keypoints": keypoints,
                    "scores": scores,
                    "detected": True,
                    "person_bbox": person_bbox,
                })
    finally:
        cap.release()

    # FPS resampling
    total_raw = len(raw_frames)
    if abs(source_fps - target_fps) < 0.5:
        resampled = raw_frames
    else:
        output_frame_count = round(total_raw * target_fps / source_fps)
        resampled = []
        for i in range(output_frame_count):
            src_idx = min(round(i * source_fps / target_fps), total_raw - 1)
            frame = dict(raw_frames[src_idx])
            frame["frame_idx"] = i
            resampled.append(frame)
        print(f"[POSE] Resampled {total_raw} frames at {source_fps:.1f}fps -> {len(resampled)} frames at {target_fps}fps")

    # Detection rate logging
    detected_count = sum(1 for f in resampled if f["detected"])
    total = len(resampled)
    rate = detected_count / total if total > 0 else 0.0
    print(f"[POSE] Detection rate: {rate:.1%} ({detected_count}/{total} frames)")
    if rate < config["detection_rate_min"]:
        print(f"[POSE] WARNING: Detection rate below threshold ({config['detection_rate_min']:.0%})")

    return resampled


def keypoints_to_joint_angles(keypoints: list[dict], config: dict) -> list[dict]:
    """Convert per-frame 2D keypoints to per-frame G1 joint angles in radians."""

    def lm(idx: int, kps: list[list[float]]) -> np.ndarray:
        """Return [x, y, z] for landmark index idx."""
        return np.array(kps[idx][:3], dtype=np.float64)

    def unit(v: np.ndarray) -> np.ndarray:
        """Return unit vector of v, or zero vector if norm is near zero."""
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros(3)

    def vec_angle(a: np.ndarray, b: np.ndarray) -> float:
        """Angle in radians between vectors a and b, in [0, pi]."""
        dot = np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0)
        return float(np.arccos(dot))

    def shoulder_angles(
        arm_vec: np.ndarray,
        side: str,
        torso_up: np.ndarray,
        torso_right: np.ndarray,
        torso_fwd: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Decompose arm_vec (shoulder->elbow) into pitch, roll, yaw in the torso frame.
        side is 'left' or 'right' -- affects roll sign convention.
        Returns (pitch, roll, yaw) in radians.
        """
        arm_u = unit(arm_vec)
        pitch = float(np.arcsin(np.clip(np.dot(arm_u, torso_fwd), -1.0, 1.0)))
        roll_raw = float(np.arcsin(np.clip(np.dot(arm_u, torso_up), -1.0, 1.0)))
        roll = roll_raw if side == "left" else -roll_raw
        yaw = float(np.arcsin(np.clip(np.dot(arm_u, torso_right), -1.0, 1.0)))
        return pitch, roll, yaw

    def hip_angles(
        thigh_vec: np.ndarray,
        side: str,
        torso_up: np.ndarray,
        torso_right: np.ndarray,
        torso_fwd: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Decompose thigh_vec (hip->knee) into pitch, roll, yaw in the torso frame.
        side is 'left' or 'right'.
        Returns (pitch, roll, yaw) in radians.
        """
        th_u = unit(thigh_vec)
        pitch = float(np.arcsin(np.clip(np.dot(th_u, torso_fwd), -1.0, 1.0)))
        roll_raw = float(np.arcsin(np.clip(-np.dot(th_u, torso_up), -1.0, 1.0)))
        roll = roll_raw if side == "left" else -roll_raw
        yaw = float(np.arcsin(np.clip(np.dot(th_u, torso_right), -1.0, 1.0)))
        return pitch, roll, yaw

    output_frames: list[dict] = []

    for frame in keypoints:
        detected = frame["detected"]
        if not detected:
            output_frames.append(
                {
                    "frame_idx": frame["frame_idx"],
                    "joint_positions": clamp_joint_angles({}),
                    "detected": detected,
                }
            )
            continue

        kps = frame["keypoints"]
        l_sh = lm(11, kps)
        r_sh = lm(12, kps)
        l_el = lm(13, kps)
        r_el = lm(14, kps)
        l_wr = lm(15, kps)
        r_wr = lm(16, kps)
        l_hip = lm(23, kps)
        r_hip = lm(24, kps)
        l_kn = lm(25, kps)
        r_kn = lm(26, kps)
        l_an = lm(27, kps)
        r_an = lm(28, kps)

        hip_mid = (l_hip + r_hip) / 2.0
        sh_mid = (l_sh + r_sh) / 2.0
        torso_up = unit(sh_mid - hip_mid)
        torso_right = unit(r_hip - l_hip)
        torso_fwd = unit(np.cross(torso_up, torso_right))

        l_upper_arm = l_el - l_sh
        l_forearm = l_wr - l_el
        kLeftElbow = -vec_angle(l_upper_arm, l_forearm)

        r_upper_arm = r_el - r_sh
        r_forearm = r_wr - r_el
        kRightElbow = vec_angle(r_upper_arm, r_forearm)

        l_thigh = l_kn - l_hip
        l_shin = l_an - l_kn
        kLeftKnee = vec_angle(l_thigh, l_shin)

        r_thigh = r_kn - r_hip
        r_shin = r_an - r_kn
        kRightKnee = vec_angle(r_thigh, r_shin)

        l_arm_vec = l_el - l_sh
        kLeftShoulderPitch, kLeftShoulderRoll, kLeftShoulderYaw = shoulder_angles(
            l_arm_vec, "left", torso_up, torso_right, torso_fwd
        )

        r_arm_vec = r_el - r_sh
        kRightShoulderPitch, kRightShoulderRoll, kRightShoulderYaw = shoulder_angles(
            r_arm_vec, "right", torso_up, torso_right, torso_fwd
        )

        kLeftWristRoll = kLeftWristPitch = kLeftWristYaw = 0.0
        kRightWristRoll = kRightWristPitch = kRightWristYaw = 0.0

        kLeftHipPitch, kLeftHipRoll, kLeftHipYaw = hip_angles(
            l_thigh, "left", torso_up, torso_right, torso_fwd
        )

        kRightHipPitch, kRightHipRoll, kRightHipYaw = hip_angles(
            r_thigh, "right", torso_up, torso_right, torso_fwd
        )

        l_heel = lm(29, kps)
        r_heel = lm(30, kps)
        l_foot_idx = lm(31, kps)
        r_foot_idx = lm(32, kps)

        l_foot_vec = l_foot_idx - l_heel
        r_foot_vec = r_foot_idx - r_heel

        kLeftAnklePitch = vec_angle(l_shin, l_foot_vec) - (np.pi / 2.0)
        kRightAnklePitch = vec_angle(r_shin, r_foot_vec) - (np.pi / 2.0)

        kLeftAnkleRoll = float(
            np.arcsin(np.clip(np.dot(unit(l_foot_vec), torso_right), -1.0, 1.0))
        )
        kRightAnkleRoll = float(
            np.arcsin(np.clip(np.dot(unit(r_foot_vec), torso_right), -1.0, 1.0))
        )

        kWaistPitch = float(np.arcsin(np.clip(np.dot(torso_up, torso_fwd), -1.0, 1.0)))
        kWaistRoll = float(np.arcsin(np.clip(np.dot(torso_up, torso_right), -1.0, 1.0)))
        kWaistYaw = float(
            np.arcsin(np.clip(np.dot(torso_fwd, np.array([1.0, 0.0, 0.0])), -1.0, 1.0))
        )

        raw_joints = {
            "kLeftShoulderPitch": kLeftShoulderPitch,
            "kLeftShoulderRoll": kLeftShoulderRoll,
            "kLeftShoulderYaw": kLeftShoulderYaw,
            "kLeftElbow": kLeftElbow,
            "kLeftWristRoll": 0.0,
            "kLeftWristPitch": 0.0,
            "kLeftWristYaw": 0.0,
            "kRightShoulderPitch": kRightShoulderPitch,
            "kRightShoulderRoll": kRightShoulderRoll,
            "kRightShoulderYaw": kRightShoulderYaw,
            "kRightElbow": kRightElbow,
            "kRightWristRoll": 0.0,
            "kRightWristPitch": 0.0,
            "kRightWristYaw": 0.0,
            "kLeftHipPitch": kLeftHipPitch,
            "kLeftHipRoll": kLeftHipRoll,
            "kLeftHipYaw": kLeftHipYaw,
            "kLeftKnee": kLeftKnee,
            "kLeftAnklePitch": kLeftAnklePitch,
            "kLeftAnkleRoll": kLeftAnkleRoll,
            "kRightHipPitch": kRightHipPitch,
            "kRightHipRoll": kRightHipRoll,
            "kRightHipYaw": kRightHipYaw,
            "kRightKnee": kRightKnee,
            "kRightAnklePitch": kRightAnklePitch,
            "kRightAnkleRoll": kRightAnkleRoll,
            "kWaistYaw": kWaistYaw,
            "kWaistRoll": kWaistRoll,
            "kWaistPitch": kWaistPitch,
        }

        output_frames.append(
            {
                "frame_idx": frame["frame_idx"],
                "joint_positions": clamp_joint_angles(raw_joints),
                "detected": detected,
            }
        )

    smoothing_cfg = config.get("smoothing", {})
    if smoothing_cfg.get("enabled", True):
        window = smoothing_cfg.get("window_frames", 5)
        poly = smoothing_cfg.get("poly_order", 3)
        if window % 2 == 0:
            window += 1
        if window <= poly:
            window = poly + 2 if (poly + 2) % 2 != 0 else poly + 3
        n_frames = len(output_frames)
        if n_frames >= window:
            for joint in JOINT_KEYS:
                series = np.array([f["joint_positions"][joint] for f in output_frames])
                smoothed = savgol_filter(series, window_length=window, polyorder=poly)
                for i, out_frame in enumerate(output_frames):
                    out_frame["joint_positions"][joint] = float(
                        np.clip(smoothed[i], JOINT_LIMITS[joint][0], JOINT_LIMITS[joint][1])
                    )
        else:
            print(
                f"[POSE] WARNING: Too few frames ({n_frames}) for smoothing "
                f"window ({window}), skipping"
            )

    print(
        f"[POSE] Angle conversion complete: {len(output_frames)} frames, "
        f"{sum(f['detected'] for f in output_frames)} detected"
    )
    return output_frames


def clamp_joint_angles(frame_joints: dict[str, float]) -> dict[str, float]:
    """Clamp all joint values to JOINT_LIMITS. Returns new dict."""
    result: dict[str, float] = {}
    for joint, (lo, hi) in JOINT_LIMITS.items():
        value = frame_joints.get(joint, 0.0)
        result[joint] = float(np.clip(value, lo, hi))
    return {k: result[k] for k in JOINT_KEYS}


def build_g1_json(dance_name: str, frames: list[dict], fps: int = 30) -> dict:
    """Assemble frames into the MotionSwitcherClient JSON schema."""
    return {
        "MotionSwitcherClient": {
            "name": dance_name,
            "fps": fps,
            "frames": [
                {
                    "time": round(i / fps, 6),
                    "joint_positions": frame["joint_positions"],
                }
                for i, frame in enumerate(frames)
            ],
        }
    }


def render_pose_video(video_path: Path, keypoints: list[dict], output_path: Path) -> None:
    """Render skeleton overlay onto original video and write to output_path."""
    SKELETON_CONNECTIONS = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (12, 24),
        (24, 26),
        (26, 28),
    ]
    LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    target_fps = 30.0

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = target_fps
    writer: cv2.VideoWriter | None = None
    temp_video = output_path.with_suffix(".temp_noaudio.mp4")

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, source_fps, (width, height))

        for i in tqdm(
            range(total_frames),
            desc="[POSE] Rendering pose video",
            unit="frame",
        ):
            ok, frame = cap.read()
            if not ok:
                break

            if keypoints:
                kp_idx = min(round(i * target_fps / source_fps), len(keypoints) - 1)
                if keypoints[kp_idx]["detected"]:
                    kps = keypoints[kp_idx]["keypoints"]
                    for a, b in SKELETON_CONNECTIONS:
                        pt_a = (int(kps[a][0] * width), int(kps[a][1] * height))
                        pt_b = (int(kps[b][0] * width), int(kps[b][1] * height))
                        cv2.line(frame, pt_a, pt_b, (0, 255, 0), 2)
                    for idx in LANDMARK_INDICES:
                        px = int(kps[idx][0] * width)
                        py = int(kps[idx][1] * height)
                        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Mux audio from source clip (OpenCV VideoWriter has no audio support)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_video),
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        _ = result
        print(f"[POSE] Pose video written (with audio): {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        shutil.move(str(temp_video), str(output_path))
        print(f"[POSE] Pose video written (no audio): {output_path}")
        if isinstance(exc, FileNotFoundError):
            print("[POSE] WARNING: ffmpeg not found, audio not muxed")
        else:
            print("[POSE] WARNING: ffmpeg audio mux failed, saved video without audio")
    finally:
        if temp_video.exists():
            temp_video.unlink()


def main() -> None:
    """CLI entry point. Parses --dance and --clip args, runs full Phase 1 pipeline."""
    parser = argparse.ArgumentParser(description="Extract pose keypoints from a dance clip.")
    parser.add_argument("--dance", required=True, help="Dance folder name (e.g. tbh_partynextdoor)")
    parser.add_argument("--clip", required=True, help="Clip filename in raw/ (e.g. clip_001.mp4)")
    args = parser.parse_args()

    config = load_config()

    clip_name = Path(args.clip).stem
    dance_name = args.dance
    raw_video = Path("dances") / dance_name / "raw" / args.clip
    if not raw_video.exists():
        print(f"[POSE] ERROR: {raw_video} not found")
        raise SystemExit(1)

    paths = setup_output_dirs(dance_name, clip_name)

    shutil.copy2(raw_video, paths["original_video"])
    print(f"[POSE] Copied original to {paths['original_video']}")

    estimator = config.get("pose_estimator", "mediapipe")
    if estimator == "vitpose":
        try:
            raw_keypoints = extract_keypoints_vitpose(raw_video, config)
        except (ImportError, NotImplementedError):
            print("[POSE] ViTPose unavailable, falling back to MediaPipe")
            config["pose_estimator"] = "mediapipe"
            raw_keypoints = extract_keypoints_mediapipe(raw_video, config)
    else:
        raw_keypoints = extract_keypoints_mediapipe(raw_video, config)
    print(f"[POSE] Extractor used: {config['pose_estimator']}")

    angle_frames = keypoints_to_joint_angles(raw_keypoints, config)

    g1_data = build_g1_json(dance_name, angle_frames, fps=config["target_fps"])

    with paths["keyframes_json"].open("w") as f:
        json.dump(g1_data, f, indent=2)
    print(f"[POSE] Keyframes written: {paths['keyframes_json']}")

    detected_count = sum(1 for f in raw_keypoints if f["detected"])
    total_frames = len(raw_keypoints)
    meta = {
        "dance_name": dance_name,
        "clip_name": clip_name,
        "pose_estimator": config["pose_estimator"],
        "total_frames": total_frames,
        "detected_frames": detected_count,
        "detection_rate": round(detected_count / total_frames, 4) if total_frames > 0 else 0.0,
        "fps": config["target_fps"],
        "flagged": (detected_count / total_frames < config["detection_rate_min"])
        if total_frames > 0
        else True,
    }
    with paths["meta_json"].open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[POSE] Meta written: {paths['meta_json']}")

    render_pose_video(raw_video, raw_keypoints, paths["pose_video"])

    print(f"[POSE] Done. Output: {paths['processed_dir']}")


if __name__ == "__main__":
    main()

if False:
    # Phase 1a smoke tests
    cfg = load_config()
    assert cfg["target_fps"] == 30
    paths = setup_output_dirs("test_dance", "clip_001")
    assert paths["keyframes_json"].parent.exists()
    fake_frames = [{"joint_positions": {k: 0.1 for k in JOINT_KEYS}} for _ in range(10)]
    g1 = build_g1_json("test_dance", fake_frames)
    assert g1["MotionSwitcherClient"]["fps"] == 30
    assert len(g1["MotionSwitcherClient"]["frames"]) == 10
    assert g1["MotionSwitcherClient"]["frames"][1]["time"] == round(1/30, 6)
    clamped = clamp_joint_angles({"kLeftElbow": -99.0, "kRightElbow": 99.0})
    assert clamped["kLeftElbow"] == -1.57
    assert clamped["kRightElbow"] == 1.57
    assert len(clamped) == 29

    # Phase 1c smoke test -- synthetic keypoints
    # Build a fake keypoints list with a T-pose (all landmarks at known positions)
    def make_fake_kp_frame(frame_idx: int) -> dict:
        kps = [[0.0, 0.0, 0.0, 1.0]] * 33
        # Place landmarks in a rough T-pose in normalized coords
        kps[11] = [0.4, 0.4, 0.0, 1.0]   # left shoulder
        kps[12] = [0.6, 0.4, 0.0, 1.0]   # right shoulder
        kps[13] = [0.3, 0.5, 0.0, 1.0]   # left elbow
        kps[14] = [0.7, 0.5, 0.0, 1.0]   # right elbow
        kps[15] = [0.2, 0.6, 0.0, 1.0]   # left wrist
        kps[16] = [0.8, 0.6, 0.0, 1.0]   # right wrist
        kps[23] = [0.45, 0.65, 0.0, 1.0] # left hip
        kps[24] = [0.55, 0.65, 0.0, 1.0] # right hip
        kps[25] = [0.45, 0.8, 0.0, 1.0]  # left knee
        kps[26] = [0.55, 0.8, 0.0, 1.0]  # right knee
        kps[27] = [0.45, 0.95, 0.0, 1.0] # left ankle
        kps[28] = [0.55, 0.95, 0.0, 1.0] # right ankle
        kps[29] = [0.44, 0.97, 0.0, 1.0] # left heel
        kps[30] = [0.54, 0.97, 0.0, 1.0] # right heel
        kps[31] = [0.46, 0.99, 0.0, 1.0] # left foot index
        kps[32] = [0.56, 0.99, 0.0, 1.0] # right foot index
        return {
            "frame_idx": frame_idx,
            "keypoints": kps,
            "scores": [1.0] * 33,
            "detected": True,
            "person_bbox": [0.2, 0.3, 0.8, 1.0],
        }

    fake_kp_frames = [make_fake_kp_frame(i) for i in range(10)]
    cfg = load_config()
    angle_frames = keypoints_to_joint_angles(fake_kp_frames, cfg)
    assert len(angle_frames) == 10
    assert all(len(f["joint_positions"]) == 29 for f in angle_frames)
    assert all(
        JOINT_LIMITS[k][0] <= v <= JOINT_LIMITS[k][1]
        for f in angle_frames
        for k, v in f["joint_positions"].items()
    ), "Joint angle out of limits"
    print("Phase 1c smoke test passed")

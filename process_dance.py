"""Phase 2 -- batch pose extraction across all clips in a dance folder."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def find_unprocessed_clips(dance_name: str) -> list[Path]:
    """Return list of .mp4 paths in raw/ that have no matching processed/ folder yet."""
    raw_dir = Path("dances") / dance_name / "raw"
    processed_dir = Path("dances") / dance_name / "processed"
    if not raw_dir.exists():
        print(f"[BATCH] ERROR: raw/ not found: {raw_dir}")
        raise FileNotFoundError(str(raw_dir))

    clips = sorted(raw_dir.glob("*.mp4"))
    unprocessed: list[Path] = []
    for clip in clips:
        clip_name = clip.stem
        out_dir = processed_dir / clip_name
        keyframes = out_dir / f"{clip_name}_keyframes.json"
        if not keyframes.exists():
            unprocessed.append(clip)
    return unprocessed


def run_extraction(dance_name: str, clip_path: Path) -> dict:
    """Run extract_pose.py for one clip, return result summary dict."""
    clip_name = clip_path.stem
    result: dict = {
        "clip_name": clip_name,
        "status": "failed",
        "total_frames": 0,
        "detection_rate": 0.0,
        "flagged": True,
    }

    proc = subprocess.run(
        [sys.executable, "extract_pose.py", "--dance", dance_name, "--clip", clip_path.name],
        capture_output=False,
    )
    if proc.returncode != 0:
        result["status"] = "failed"
        return result

    meta_path = Path("dances") / dance_name / "processed" / clip_name / f"{clip_name}_meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        result["total_frames"] = meta.get("total_frames", 0)
        result["detection_rate"] = meta.get("detection_rate", 0.0)
        result["flagged"] = meta.get("flagged", True)
        result["status"] = "ok"
    else:
        result["status"] = "failed"

    return result


def print_summary_table(results: list[dict]) -> None:
    """Print formatted table of clip name, frame count, detection rate, status."""
    if not results:
        print("[BATCH] No clips processed.")
        return

    name_w = max(len(r["clip_name"]) for r in results)
    name_w = max(name_w, len("clip"))
    header = f"{'clip':<{name_w}}  {'frames':>6}  {'detect%':>7}  {'flagged':>7}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        rate_pct = f"{r['detection_rate'] * 100:.1f}%"
        flagged = "yes" if r["flagged"] else "no"
        print(
            f"{r['clip_name']:<{name_w}}  {r['total_frames']:>6}  "
            f"{rate_pct:>7}  {flagged:>7}  {r['status']}"
        )


def main() -> None:
    """CLI entry point. Parses --dance arg, runs batch extraction, prints summary."""
    parser = argparse.ArgumentParser(description="Batch extract pose from all clips in a dance.")
    parser.add_argument("--dance", required=True, help="Dance folder name (e.g. TAKA LA DENTRO)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all clips, including ones already in processed/",
    )
    args = parser.parse_args()

    dance_name = args.dance
    raw_dir = Path("dances") / dance_name / "raw"

    if args.force:
        clips = sorted(raw_dir.glob("*.mp4")) if raw_dir.exists() else []
    else:
        clips = find_unprocessed_clips(dance_name)

    if not clips:
        print(f"[BATCH] No clips to process in {raw_dir}")
        return

    print(f"[BATCH] Processing {len(clips)} clip(s) for {dance_name}")
    results: list[dict] = []
    for clip_path in tqdm(clips, desc="[BATCH] Clips", unit="clip"):
        print(f"[BATCH] Starting: {clip_path.name}")
        results.append(run_extraction(dance_name, clip_path))

    print()
    print_summary_table(results)
    failed = sum(1 for r in results if r["status"] != "ok")
    print(f"[BATCH] Done. {len(results) - failed}/{len(results)} succeeded.")


if __name__ == "__main__":
    main()

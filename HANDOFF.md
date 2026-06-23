# Handoff: otto-dance-pipeline

**Date:** 2026-06-23  
**From:** Cursor agent (scaffold session)  
**To:** Claude (next session) — read this file and respond with concrete next steps for the implementing agent.

---

## Project goal

Convert TikTok dance videos into Unitree G1 humanoid robot motion files.

Pipeline: `MP4` → pose estimation → G1 joint angles (radians) → `MotionSwitcherClient` JSON.

Authoritative project context lives in [`.cursorrules`](.cursorrules). Do not modify `.cursorrules` unless the user asks.

---

## What was completed (scaffold only)

The initial repo scaffold is done. **No pipeline logic is implemented yet** — every function body is `pass`.

### Files created

| File | Purpose |
|------|---------|
| [`pipeline_config.yaml`](pipeline_config.yaml) | Tunable constants (pose estimator, thresholds, smoothing, angle conversion) |
| [`requirements.txt`](requirements.txt) | Dependencies (opencv, numpy, scipy, tqdm, pyyaml, mediapipe, mmpose, mmcv, torch, torchvision) |
| [`.gitignore`](.gitignore) | Ignores `dances/`, pycache, `.env`, build artifacts |
| [`README.md`](README.md) | Usage, folder layout, phases, prep conventions |
| [`extract_pose.py`](extract_pose.py) | Phase 1 stubs + `JOINT_LIMITS` / `JOINT_KEYS` (29 joints) |
| [`process_dance.py`](process_dance.py) | Phase 2 batch runner stubs |
| [`average_dance.py`](average_dance.py) | Phase 3 averaging stubs |
| [`dances/.gitkeep`](dances/.gitkeep) | Preserves empty `dances/` dir in git |

### Stub inventory

**`extract_pose.py`** — functions to implement next:
- `load_config`, `setup_output_dirs`
- `extract_keypoints_vitpose`, `extract_keypoints_mediapipe`
- `keypoints_to_joint_angles`, `clamp_joint_angles`, `build_g1_json`
- `render_pose_video`, `main`

**`process_dance.py`** — `find_unprocessed_clips`, `run_extraction`, `print_summary_table`, `main`

**`average_dance.py`** — `load_keyframe_files`, `align_clip_lengths`, `circular_mean_frames`, `compute_confidence`, `main`

### Verification done

- `python3 -m py_compile extract_pose.py process_dance.py average_dance.py` passes
- `.cursorrules` was not touched

---

## Design decisions agreed with user

### Dance folder layout

```text
dances/
  {dance_name}/          # lowercase snake_case, matches --dance CLI flag
    raw/                 # user creates; drops trimmed .mp4 clips here
    processed/           # pipeline creates
      {clip_name}/
        {clip_name}_original.mp4
        {clip_name}_pose.mp4
        {clip_name}_keyframes.json
    averaged/            # pipeline creates
      {dance_name}_averaged.json
      {dance_name}_confidence.json
```

### Clip prep (user is doing this now)

- **5 clips per dance** is the practical target; **3 minimum** for averaging (`min_clips_for_average` in config)
- **Manual trim** before upload: frame 0 = first dance move (skip camera-smile preambles). Preferred over per-clip offset config for v1
- **Clip filenames:** downloaded TikTok names are fine; no required `take_01` naming
- **Time alignment:** all clips for a dance must start at the same choreographic moment (frame 0 = same beat/move)
- **Multi-person clips:** OK. Phase 1 must select **one** dancer per clip (largest/most central body or highest mean keypoint confidence) and track only that person

### Deferred (do not build unless user asks)

- **Music / audio sync** with robot playback (Phase 4+). Motion JSON uses `time` at 30 fps only
- **Per-clip trim.yaml** start offsets (optional future if user keeps untrimmed downloads)
- **Auto motion-start detection**
- **Phases 4–6:** Isaac Sim, adjustment loop, real robot deployment

### Pose estimator priority

1. ViTPose via mmpose (`pipeline_config.yaml` → `pose_estimator: vitpose`)
2. MediaPipe Holistic fallback

### Style rules (from `.cursorrules`)

- Python 3.10+, type hints on all signatures
- Print prefixes: `[POSE]`, `[BATCH]`, `[AVG]`
- tqdm in frame loops, no print spam
- All tunables from `pipeline_config.yaml`, never hardcoded
- f-strings only; no em dashes in comments/docstrings

### Target output JSON schema

```json
{
  "MotionSwitcherClient": {
    "name": "<dance_name>",
    "fps": 30,
    "frames": [
      {
        "time": 0.0,
        "joint_positions": {
          "kLeftHipPitch": 0.0
        }
      }
    ]
  }
}
```

All 29 joint keys must appear in every frame. Values in radians, clamped to `JOINT_LIMITS` in `extract_pose.py`.

---

## User state

- User is **normalizing/trimming clips now** and dropping them into `dances/{dance_name}/raw/`
- Dance folders may exist locally under `dances/` (gitignored). Recommend `snake_case` names for CLI ergonomics
- User wants Phase 1 implemented next so they can get **`_pose.mp4` overlays** and **`_keyframes.json`** files
- Packages are **not installed** yet (`pip install -r requirements.txt` still needed)

---

## What is NOT done

- [ ] No function bodies implemented
- [ ] No `pip install` run
- [ ] No end-to-end test on real clips
- [ ] `process_dance.py` and `average_dance.py` untouched

---

## Request for Claude

Read this handoff plus [`.cursorrules`](.cursorrules), [`pipeline_config.yaml`](pipeline_config.yaml), and the three Python stubs.

**Respond with:**

1. **Recommended implementation order** for Phase 1 (`extract_pose.py`) — which functions to implement first and why
2. **ViTPose / mmpose setup notes** — install pitfalls, model weights, minimum code path to get keypoints from one test clip
3. **Multi-person selection approach** — concrete strategy for ViTPose and MediaPipe fallback
4. **Bone-vector angle conversion** — high-level math plan matching `angle_conversion.method: bone_vector` and `reference_frame: torso` in config
5. **Definition of done** for Phase 1 — what files should exist after `python extract_pose.py --dance <name> --clip <filename>` on one clip
6. **Suggested follow-up order** for Phase 2 and Phase 3 after Phase 1 works

Keep next steps actionable for the Cursor implementing agent. Do not implement code in your response — plan only.

# otto-dance-pipeline

Converts TikTok dance videos into Unitree G1 humanoid robot motion files.

## Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Extract pose from a single clip:

```bash
python extract_pose.py --dance <name> --clip <filename>
```

Batch process all clips in a dance folder:

```bash
python process_dance.py --dance <name>
```

Average processed clips into canonical motion:

```bash
python average_dance.py --dance <name> --min-clips 3
```

## Output

Each dance lives under `dances/{dance_name}/`:

```text
dances/
  {dance_name}/
    raw/                          # drop .mp4 clips here (you create this)
      take_01.mp4
    processed/                    # created by Phase 1 / 2
      {clip_name}/
        {clip_name}_original.mp4
        {clip_name}_pose.mp4
        {clip_name}_keyframes.json
    averaged/                     # created by Phase 3
      {dance_name}_averaged.json
      {dance_name}_confidence.json
```

### Dance folder conventions

- **Dance name:** lowercase `snake_case`, no spaces (e.g. `renegade`, `say_so`). Must match the `--dance` CLI flag.
- **Clip files:** any `.mp4` filename in `raw/`; the stem becomes `clip_name`. Downloaded TikTok names are fine.
- **Recommended:** 5 aligned clips per dance (minimum 3 for averaging).
- **Trim clips** so frame 0 is the first dance move (skip camera-smile preambles).
- **Multi-person clips:** OK. Phase 1 will select and track one dancer per clip.

## Phases

| Phase | Script | Status |
|-------|--------|--------|
| 1 | `extract_pose.py` | Single video to keyframes JSON + pose overlay |
| 2 | `process_dance.py` | Batch runner over all clips in a dance folder |
| 3 | `average_dance.py` | Circular mean across clips to canonical motion |
| 4 | Isaac Sim playback | Deferred |
| 5 | Config-driven adjustment loop | Deferred |
| 6 | Real robot deployment | Deferred |

# Praxis Motion Intelligence

Praxis Motion Intelligence is a self-contained Python prototype for human action analysis in rehabilitation settings. It accepts video uploads or landmark JSON, estimates movement quality from pose trajectories, compares joint-angle patterns against UCF101-inspired reference templates, flags movement limitations using stroke-rehabilitation heuristics, and recommends guided exercises for physiotherapists.

## What it does

- Ingests video files and extracts metadata with `ffprobe`
- Supports plug-in pose estimation backends
- Falls back to landmark JSON or demo pose sequences when no vision model is installed
- Computes clinically relevant joint angles and range-of-motion scores
- Compares actions against reference trajectories derived from representative UCF101 classes
- Detects possible motor limitations using stroke-informed asymmetry and ROM rules
- Generates quantitative feedback and visual rehab guidance in a browser report

## Current environment note

The local environment does not include `opencv` or `mediapipe`, so the repository ships with:

- a complete analysis engine
- an optional `MediaPipePoseEstimator` hook that activates automatically if dependencies are installed later
- demo/reference landmark sequences so the full workflow can run now

## Run

```bash
python3 app.py
```

Then open `http://127.0.0.1:8000`.

## Input modes

- Upload a video file: metadata is extracted immediately; if a pose backend is available it is used
- Upload landmark JSON: runs the full analysis pipeline directly
- Run built-in demos: useful for testing the full report flow without external dependencies

## Landmark JSON format

```json
{
  "label": "demo_walking",
  "fps": 24,
  "frames": [
    {
      "timestamp": 0.0,
      "landmarks": {
        "left_shoulder": {"x": 0.45, "y": 0.30},
        "left_elbow": {"x": 0.42, "y": 0.44},
        "left_wrist": {"x": 0.40, "y": 0.58}
      }
    }
  ]
}
```

Expected landmark names follow common pose-estimation conventions:

`nose`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

## Project layout

- `app.py`: launch script
- `src/praxis_ai/server.py`: HTTP server and upload flow
- `src/praxis_ai/analysis.py`: joint-angle scoring and reference comparison
- `src/praxis_ai/pose_estimation.py`: video and pose-estimation adapters
- `src/praxis_ai/rehab.py`: stroke-informed impairment detection and exercise plans
- `data/reference_patterns.json`: UCF101-inspired movement references
- `data/demo_landmarks/*.json`: sample pose sequences

## Extending with a real pose model

If `mediapipe` and `opencv-python` are installed in a richer environment, the server will automatically attempt real video pose extraction before using fallbacks.

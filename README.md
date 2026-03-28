# Praxis Motion Intelligence

Praxis Motion Intelligence is a rehab and recovery focused movement analysis system with a Python backend and React frontend. It accepts video, webcam capture, or landmark JSON, extracts pose trajectories, computes joint-angle metrics, compares movement against a UCF101-derived normal baseline, and interprets deviation severity as normal, injury recovery, or neurological limitation.

## What It Does

- Accepts uploaded video, webcam recordings, or pasted landmark JSON
- Extracts pose landmarks from video with MediaPipe when a backend is available
- Computes joint-angle series for elbows, shoulders, hips, and knees
- Scores movement using mobility, symmetry, and smoothness
- Builds multi-dataset severity interpretation on top of a UCF101 baseline
- Classifies each joint as:
  - `Normal`
  - `Injury Recovery`
  - `Severe Limitation`
- Produces an overall condition summary
- Generates human-readable feedback such as reduced ROM, asymmetry, overextension, and compensation patterns
- Renders a stickman overlay video with severity-colored joints
- Shows joint charts, event timeline, repetition summary, and session comparison in the UI

## Core Architecture

Praxis uses a UCF-first reference design:

- `data/normal.json` is the primary baseline and is derived from UCF101
- `data/injury.json` models moderate limitation and is calibrated using the sports injury CSV in the project root
- `data/stroke.json` models severe neurological limitation from stroke threshold data

Comparison logic:

1. The user movement is always compared against the UCF101 normal profile.
2. The injury and stroke profiles are only used to interpret severity.
3. The system returns joint-level condition labels and an overall condition summary.

## Runtime Flow

1. The frontend sends a video, webcam clip, or landmark JSON to `POST /api/analyze`.
2. The backend extracts landmarks and builds a `PoseSequence`.
3. Joint-angle series are computed.
4. Existing scores are calculated:
   - mobility
   - symmetry
   - smoothness
5. The system compares joint behavior against `normal.json`.
6. It uses `injury.json` and `stroke.json` to classify severity.
7. The report is serialized and returned with:
   - scores
   - joint status
   - overall condition
   - feedback
   - annotations
   - joint charts
   - overlay replay

## Current Feature Set

- Video upload
- Webcam recording
- Landmark JSON input
- Condition selector:
  - Normal
  - Injury Recovery
  - Neurological Condition
- Overall movement scoring
- Joint-level condition labels
- Stickman replay overlay with green/yellow/red status coloring
- Event timeline
- Joint motion charts
- Repetition summary
- Session comparison
- PDF-friendly export via browser print

## Datasets

### UCF101

Used as the core movement reference.

- Builds `data/normal.json`
- Defines how movement should ideally be performed

### Stroke Dataset

Used to represent severe limitation.

- Processed from the stroke MATLAB threshold data already used in the repo
- Builds `data/stroke.json`

### Sports Injury CSV

Used to calibrate moderate limitation.

- File: `multimodal_sports_injury_dataset.csv`
- Consumed by `scripts/build_condition_profiles.py`
- Builds `data/injury.json`

## API

The backend is implemented in [src/praxis_ai/server.py](/home/rohitpujari/Documents/praxis/src/praxis_ai/server.py) and exposes:

- `GET /api/health`
- `POST /api/analyze`

Supported form fields:

- `video_file`
- `landmarks_json`
- `condition_profile`

## Setup

```bash
uv sync
npm install
python scripts/build_condition_profiles.py
npm run build
UV_CACHE_DIR=.uv-cache uv run praxis
```

Then open `http://127.0.0.1:8000`.

## Input Modes

- Upload a video file
- Record a webcam clip in the browser
- Paste landmark JSON directly

## Output

Each analysis response can include:

- `overall_score`
- `mobility_score`
- `symmetry_score`
- `smoothness_score`
- `overall_condition`
- `joint_status`
- `joint_deviation`
- joint summary rows
- joint charts
- annotations and event timeline entries
- repetition summary
- detected limitations
- exercise guidance
- human-readable feedback
- metadata, including dataset/profile provenance
- overlay video

## Project Layout

- `main.py`: CLI entrypoint exposed as `praxis`
- `app.py`: direct Python launch shim
- `src/praxis_ai/server.py`: HTTP server, multipart parsing, and frontend asset serving
- `src/praxis_ai/pose_estimation.py`: MediaPipe extraction, probing, and overlay rendering
- `src/praxis_ai/analysis.py`: joint-angle computation, scoring, and condition-aware interpretation
- `src/praxis_ai/reference_data.py`: loads rules, thresholds, and profile JSONs
- `src/praxis_ai/reporting.py`: API response serialization
- `src/praxis_ai/rehab.py`: limitation detection and exercise recommendation
- `frontend/`: React application source
- `data/`: checked-in runtime calibration and profile artifacts
- `scripts/`: offline processing and profile-generation utilities
- `models/`: local MediaPipe task assets when required by the installed backend

## Offline Profile Generation

Available helper scripts:

- `scripts/build_ucf_reference_stats.py`: computes UCF101 joint statistics
- `scripts/build_condition_profiles.py`: builds `normal.json`, `injury.json`, and `stroke.json`
- `scripts/extract_stroke_thresholds.py`: extracts stroke thresholds from local MATLAB datasets
- `scripts/calibrate_dataset.py`: derives calibration targets from local datasets

## Architecture Diagram

An importable draw.io architecture file is available at:

- [praxis_movement_intelligence_architecture.drawio](/home/rohitpujari/Documents/praxis/praxis_movement_intelligence_architecture.drawio)

## Pose Backend Behavior

If video landmarks cannot be extracted, the server fails clearly instead of substituting a demo result. To proceed, provide a supported video, a `.pose.json` sidecar file, or paste landmark JSON directly.

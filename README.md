# Praxis Motion Intelligence

Praxis Motion Intelligence is an AI-assisted human movement analysis system for rehab, recovery, and movement-quality assessment. It combines pose estimation, classical kinematic analysis, dataset-derived reference profiles, a lightweight baseline classifier, and an additive spatio-temporal deep-learning model.

The system is designed around one core principle:

- **UCF101 is the primary reference for normal movement**

Everything else is layered on top of that baseline.

## Current System Summary

Praxis currently includes five major layers:

1. **Pose Extraction**
   - MediaPipe extracts body landmarks from video
   - Landmark JSON is also supported as a direct input path

2. **Kinematic Analysis**
   - Joint angle computation
   - Mobility, symmetry, and smoothness scoring
   - Limitation detection and exercise guidance

3. **Reference-Profile Interpretation**
   - `data/normal.json` from UCF101
   - `data/injury.json` from the sports injury CSV plus UCF/stroke anchors
   - `data/stroke.json` from stroke threshold data

4. **Baseline Model**
   - A lightweight nearest-centroid classifier built from the three reference profiles
   - Used as an auxiliary prediction path

5. **Deep Learning Add-On**
   - ST-GCN + LSTM hybrid
   - Optional, additive, and demo-safe
   - Predicts condition from pose sequences without replacing the explainable pipeline

## What The Project Does

- Accepts uploaded video, webcam recordings, or landmark JSON
- Extracts pose with MediaPipe when available
- Builds joint-angle series for elbows, shoulders, hips, and knees
- Scores movement using:
  - mobility
  - symmetry
  - smoothness
- Compares movement against a UCF101-derived normal reference
- Interprets severity using injury and stroke profiles
- Produces:
  - joint-level condition labels
  - overall condition
  - explainable feedback
  - charts
  - event timeline
  - repetition summary
  - session comparison
  - overlay replay with a colored stickman

## Core Design Principle

Praxis uses a **UCF-first** architecture.

- `data/normal.json` defines how movement should ideally be performed
- `data/injury.json` represents moderate limitation
- `data/stroke.json` represents severe neurological limitation

Comparison logic:

1. The user is always compared to the UCF101 normal profile first.
2. Injury and stroke profiles are only used to interpret severity.
3. The output remains explainable, even when the deep model is enabled.

This keeps the system clinically intuitive and consistent with the original design.

## Model Stack

### 1. Pose Model

Praxis uses **MediaPipe** as the pretrained pose-estimation model.

Role:

- convert raw video into body landmarks
- produce a `PoseSequence`

This is the only external pretrained vision model required for the core app.

### 2. Rule and Profile-Based Intelligence Layer

This is the main reasoning layer of the current system.

Files:

- `src/praxis_ai/analysis.py`
- `src/praxis_ai/reference_data.py`
- `src/praxis_ai/rehab.py`

Role:

- compute joint angles
- compute scores
- detect limitations
- compare against normal/injury/stroke profiles
- generate feedback and joint status

### 3. Basic Condition Model

File:

- `src/praxis_ai/basic_model.py`

Artifact:

- `data/basic_condition_model.json`

Type:

- nearest-centroid profile classifier

Role:

- acts as a lightweight baseline model
- uses profile centroids from:
  - `normal.json`
  - `injury.json`
  - `stroke.json`
- returns an auxiliary condition prediction

### 4. ST-GCN + LSTM Deep Model

File:

- `src/praxis_ai/deep_learning.py`

Training script:

- `scripts/train_stgcn_transformer.py`

Checkpoint:

- `models/stgcn_transformer_demo.pt`

Type:

- spatial graph modeling over joints
- temporal modeling over frame sequences

Role:

- predicts:
  - `deep_model_prediction`
  - `deep_model_confidence`
  - `deep_joint_importance`
- remains optional
- does not replace the main analysis pipeline

## ST-GCN + LSTM Architecture

The deep-learning model treats human movement in two ways at once:

### Spatial View: Graph

Each frame is represented as a graph:

- nodes = joints
- edges = body connections

This allows the model to learn joint-to-joint spatial relationships such as:

- shoulder to elbow coordination
- hip to knee alignment
- left-right structural asymmetry

### Temporal View: Sequence

The graph outputs are processed over time with an LSTM.

This allows the model to learn movement dynamics such as:

- hesitation
- rhythm
- compensation
- motion smoothness across frames

### Outputs

The deep model returns:

- condition class:
  - `Normal`
  - `Injury Recovery`
  - `Severe Limitation`
- confidence score
- joint importance weights

### Why It Is Additive

The deep model is intentionally not the only source of truth.

Runtime architecture:

`Pose -> Feature Tensor -> Deep Model -> Condition Prediction -> Existing Feedback System`

So:

- the model predicts
- the existing logic explains

## Runtime Data Flow

1. Frontend sends video, webcam clip, or landmark JSON to `POST /api/analyze`
2. Backend extracts landmarks and builds a `PoseSequence`
3. Joint-angle series are computed
4. Core scores are computed:
   - mobility
   - symmetry
   - smoothness
5. Joint status is classified using:
   - `normal.json`
   - `injury.json`
   - `stroke.json`
6. Basic model prediction is computed
7. Deep model prediction is computed if:
   - `torch` is installed
   - a checkpoint exists
8. Existing feedback and rehab guidance are generated
9. Stickman overlay is rendered with:
   - severity colors
   - optional deep-model importance emphasis
10. Report is serialized and returned to the UI

## Input Modes

- Upload a video file
- Record a webcam clip in the browser
- Paste landmark JSON directly

## UI Features

- Video upload
- Webcam recording
- Landmark JSON input
- Condition selector:
  - Normal
  - Injury Recovery
  - Neurological Condition
- Overall movement scoring
- Joint-level condition labels
- Overall condition interpretation
- Basic model prediction display
- ST-GCN + LSTM prediction display
- Deep model attention panel
- Stickman replay overlay with:
  - Green = Normal
  - Yellow = Injury Recovery
  - Red = Severe Limitation
- Event timeline
- Joint motion charts
- Repetition summary
- Session comparison
- Browser-print PDF export

## Datasets

### UCF101

Purpose:

- primary reference for normal movement

Artifact:

- `data/normal.json`

Builder:

- `scripts/build_ucf_reference_stats.py`
- `scripts/build_condition_profiles.py`

### Stroke Dataset

Purpose:

- severe movement limitation reference

Artifacts:

- `data/stroke_thresholds.json`
- `data/stroke.json`

Builder:

- `scripts/extract_stroke_thresholds.py`
- `scripts/build_condition_profiles.py`

### Sports Injury CSV

File:

- `multimodal_sports_injury_dataset.csv`

Purpose:

- calibrates moderate injury-recovery profile

Artifact:

- `data/injury.json`

Builder:

- `scripts/build_condition_profiles.py`

## Important Runtime Artifacts

Reference artifacts:

- `data/normal.json`
- `data/injury.json`
- `data/stroke.json`

Model artifacts:

- `data/basic_condition_model.json`
- `models/stgcn_transformer_demo.pt`

Training summary:

- `data/deep_learning_training_summary.json`

Diagram:

- `praxis_movement_intelligence_architecture.drawio`

Pitch deck:

- `hackathon_pitch_deck.pdf`

## API

Backend entrypoint:

- `src/praxis_ai/server.py`

Endpoints:

- `GET /api/health`
- `POST /api/analyze`

Supported form fields:

- `video_file`
- `landmarks_json`
- `condition_profile`

## Response Fields

The analysis response can include:

- `overall_score`
- `mobility_score`
- `symmetry_score`
- `smoothness_score`
- `overall_condition`
- `joint_status`
- `joint_deviation`
- `joint_summary`
- `joint_charts`
- `annotations`
- `rep_summary`
- `feedback`
- `limitations`
- `exercises`
- `model_prediction`
- `model_distances`
- `deep_model_prediction`
- `deep_model_confidence`
- `deep_joint_importance`
- `metadata`
- `overlay_video`

## Setup

```bash
uv sync
npm install
python scripts/build_condition_profiles.py
python scripts/build_basic_condition_model.py
UV_CACHE_DIR=.uv-cache uv add torch
UV_CACHE_DIR=.uv-cache uv run python scripts/train_stgcn_transformer.py
npm run build
UV_CACHE_DIR=.uv-cache uv run praxis
```

Then open:

```bash
http://127.0.0.1:8000
```

## Training and Inference Notes

### Deep Model Training

Train the ST-GCN + LSTM checkpoint with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_stgcn_transformer.py
```

Current behavior:

- if `torch` is available, the script trains a lightweight demo checkpoint
- if `torch` is missing, it writes a summary file instead of breaking the repo

### CUDA Behavior

The deep-learning path now automatically selects:

- `cuda` if `torch.cuda.is_available()` is true
- otherwise `cpu`

That applies to:

- checkpoint loading
- inference
- training

### Optionality

The deep model is optional.

If the environment does not have:

- `torch`, or
- `models/stgcn_transformer_demo.pt`

the rest of the system still works exactly as before.

## Project Layout

- `main.py`: CLI entrypoint exposed as `praxis`
- `app.py`: direct Python launch shim
- `src/praxis_ai/server.py`: HTTP server and request handling
- `src/praxis_ai/pose_estimation.py`: MediaPipe extraction, probing, and overlay rendering
- `src/praxis_ai/analysis.py`: scoring, profile logic, feedback, and deep-model integration
- `src/praxis_ai/reference_data.py`: profile and rule loading
- `src/praxis_ai/reporting.py`: API serialization
- `src/praxis_ai/rehab.py`: limitation detection and exercise recommendation
- `src/praxis_ai/basic_model.py`: centroid-based baseline classifier
- `src/praxis_ai/deep_learning.py`: ST-GCN + Transformer module
- `frontend/`: React frontend
- `data/`: checked-in calibration, profiles, and model summaries
- `scripts/`: offline profile generation and model-training utilities
- `models/`: MediaPipe assets and deep-learning checkpoints

## How To Explain The System

Short version:

> Praxis uses MediaPipe to extract pose, compares movement to a UCF101-derived normal baseline, uses injury and stroke references to interpret severity, and adds a graph-based deep-learning model with an LSTM temporal encoder to classify movement over time. The deep model predicts, and the existing pipeline explains.

If someone asks “do you have a model?”:

> Yes. We use MediaPipe for pose estimation, a lightweight centroid baseline model for condition classification, and an additive ST-GCN + Transformer deep-learning model for spatio-temporal movement prediction.

## Additional Documentation

- `STGCN_TRANSFORMER_EXPLAINER.md`
- `HACKATHON_JUDGES_REFERENCE.md`
- `praxis_movement_intelligence_architecture.drawio`

## Failure Behavior

If pose landmarks cannot be extracted, the server fails clearly instead of substituting demo results. To proceed, provide:

- a supported video
- a `.pose.json` sidecar file
- or landmark JSON directly

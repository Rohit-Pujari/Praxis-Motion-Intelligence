# ST-GCN + Transformer Add-On

## What It Is

This project now supports an additive deep-learning path built on top of the existing movement analysis system.

The model is a lightweight hybrid:

- **ST-GCN style graph block**
  - treats the body as a graph
  - joints are nodes
  - body connections are edges
  - captures spatial structure inside each frame

- **Transformer temporal encoder**
  - treats movement as a sequence over time
  - captures dynamics such as hesitation, rhythm, compensation, and movement evolution

## Why This Fits The Project

The existing pipeline already does:

- pose extraction
- angle computation
- scoring
- explainable feedback

The deep-learning model does **not** replace that.

It only adds:

- `predicted_condition`
- `confidence_score`
- `joint_importance`

So the full runtime story becomes:

1. MediaPipe extracts landmarks
2. Pose sequence is converted to `(frames, joints, features)`
3. ST-GCN captures body structure
4. Transformer captures motion over time
5. Model predicts condition
6. Existing rule-based system explains the result

## How To Explain It In A Hackathon

Use this short version:

> “We model the body as a graph across time. The graph module learns joint-to-joint spatial relationships, and the temporal module learns how the movement evolves frame by frame. That gives us a deep-learning condition prediction, while our existing analysis layer still provides interpretable feedback.”

## Key Talking Points

- We are not only detecting pose, we are learning movement patterns.
- The graph side models biomechanics structure.
- The temporal side models dynamics over time.
- UCF101 still anchors normal movement.
- Injury and stroke data support condition-level learning.
- The deep model predicts; the existing system explains.
- This is demo-safe because it is lightweight and optional.

## Files Added

- `src/praxis_ai/deep_learning.py`
- `scripts/train_stgcn_transformer.py`

## Practical Note

If PyTorch is not installed, the main app still works.

The deep-learning module is optional and only activates when a checkpoint exists and `torch` is available.

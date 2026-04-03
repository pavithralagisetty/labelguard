# LabelGuard

A FiftyOne plugin that detects mislabeled videos using **Twelve Labs Marengo** embeddings and AI-generated descriptions. Select samples in the FiftyOne App, run the operator, and instantly see which labels don't match the actual video content.

## Why This Exists

Large video datasets ship with label noise baked in. The [Kinetics dataset](https://voxel51.com/blog/the-kinetics-dataset-train-and-evaluate-video-classification-models) (300K YouTube videos, 400 action classes) is a well-known example -- a Voxel51 engineer manually discovered a clip labeled "front crawl" that actually showed breaststroke and backstroke. The correct label didn't appear anywhere in the video. The only way they found it was by training a full model first and hoping it disagreed with the bad label.

That workflow is backwards: **you shouldn't need to train a model to find out your training data is wrong.**

LabelGuard finds these mismatches automatically with zero training required. Twelve Labs Marengo already understands video content -- we just compare what it sees against what the label claims. No model training, no waiting for convergence, no hoping the model learns enough to spot the error. Select your samples, run the operator, get results.

## Demo

![LabelGuard Demo](output_edited.gif)

## How It Works

LabelGuard uses a dual verification approach powered by Twelve Labs:

1. **Embedding Similarity** -- Marengo embeds the video and the ground truth label text into the same 1024-dim vector space. Cosine similarity below a threshold flags a mismatch.
2. **AI Description** -- Twelve Labs Analyze generates a one-sentence description of the video content, displayed as an overlay so you can visually compare it against the ground truth label.

```
Video file  --> Marengo video embedding (1024-dim)
Label text  --> Marengo text embedding  (1024-dim)
cosine_similarity(video, text) --> MATCH or MISMATCH

Video file  --> Twelve Labs Analyze --> AI-generated description
```

## Output Fields

After running the operator, each sample gets these fields:

| Field | Type | Description |
|---|---|---|
| `similarity_score` | float | Cosine similarity between video and label embeddings (-1 to 1) |
| `label_check` | string | `"MATCH"` or `"MISMATCH"` |
| `is_mislabeled` | boolean | `True` if similarity is below threshold |
| `video_description` | Classification | AI-generated description of the video content (shown as overlay) |

## Installation

### 1. Clone this repo into your FiftyOne plugins directory

```bash
cd ~/fiftyone/__plugins__
git clone https://github.com/pavithralagisetty/labelguard.git
```

Or copy manually:

```bash
mkdir -p ~/fiftyone/__plugins__/labelguard
cp __init__.py fiftyone.yml ~/fiftyone/__plugins__/labelguard/
```

### 2. Install dependencies

```bash
pip install twelvelabs numpy
```

### 3. Set your Twelve Labs API key

```bash
export TWELVELABS_API_KEY=your_key_here
```

### 4. Verify the plugin is registered

```bash
fiftyone plugins list
fiftyone operators list
```

You should see `@pavithralagisetty/labelguard` and `check_video_label`.

## Usage

### In the FiftyOne App

1. Launch FiftyOne with your video dataset
2. Select one or more video samples in the grid
3. Press `` ` `` (backtick) to open the operator menu
4. Search for **"Check Video Label with Twelve Labs"**
5. Set the label field (default: `ground_truth`)
6. Click **Execute**

The operator will process each selected sample and display progress. When done, you'll see:
- The ground truth label and AI description overlaid on each sample
- `similarity_score`, `label_check`, and `is_mislabeled` fields in the sidebar

### Filtering Results

```python
from fiftyone import ViewField as F

# View only mislabeled samples
mislabeled = dataset.match(F("is_mislabeled") == True)
session.view = mislabeled

# Filter by similarity score
low_sim = dataset.match(F("similarity_score") < 0.5)
session.view = low_sim

# Sort by most suspicious first
suspicious = dataset.sort_by("similarity_score")
session.view = suspicious
```

## Twelve Labs Integration

This plugin uses two Twelve Labs capabilities:

- **Marengo Embed API** (`embed.tasks.create` + `embed.create`) -- Generates joint video-text embeddings for cosine similarity comparison. The video embed is async (requires polling), the text embed is synchronous.
- **Analyze API** (`tl_client.analyze`) -- Generates a natural language description of the video content, giving users an interpretable explanation alongside the numeric score.

## Project Structure

```
labelguard/
├── __init__.py          # Plugin logic (operator + Twelve Labs integration)
├── fiftyone.yml         # FiftyOne plugin manifest
├── requirements.txt     # Python dependencies
└── README.md
```

## Tech Stack

- [FiftyOne](https://docs.voxel51.com/) -- Open-source dataset curation and visualization
- [Twelve Labs Marengo](https://docs.twelvelabs.io/) -- Video-language embedding model
- [Twelve Labs Analyze](https://docs.twelvelabs.io/) -- Video understanding and description

## Submission

Built for the [Video Understanding AI Hackathon](https://github.com/harpreetsahota204/video_understanding_hackathon) (April 2026).

---
marp: true
theme: uncover
paginate: true
backgroundColor: #1a1a2e
color: #eaeaea
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #e94560;
  }
  h2 {
    color: #0f3460;
    background: #e94560;
    padding: 4px 16px;
    border-radius: 8px;
    display: inline-block;
  }
  strong {
    color: #e94560;
  }
  code {
    background: #16213e;
    color: #00d2ff;
  }
  table {
    font-size: 0.8em;
  }
  th {
    background: #e94560;
    color: white;
  }
  td {
    background: #16213e;
  }
  blockquote {
    border-left: 4px solid #e94560;
    background: #16213e;
    padding: 12px 20px;
    font-style: italic;
  }
---

# LabelGuard

### Find mislabeled videos — zero training required

A FiftyOne plugin powered by **Twelve Labs Marengo**

---

## The Problem

Large video datasets ship with **label noise baked in**

> A clip in Kinetics labeled **"front crawl"** actually showed breaststroke and backstroke. The correct label didn't appear anywhere in the video.

**Current solution:** Train a model first, hope it disagrees with the bad label

That's **expensive**, **slow**, and **unreliable**

---

## Our Solution

**Skip the training. Ask the video directly.**

Twelve Labs Marengo embeds video and text into the **same vector space**

```
Video file  →  Marengo video embedding (1024-dim)
Label text  →  Marengo text embedding  (1024-dim)

cosine_similarity(video, text)  →  MATCH or MISMATCH
```

If they don't match → **the label is wrong**

---

## But Wait, There's More

We don't just give you a number — we show you **why**

Twelve Labs Analyze generates a **one-sentence description** of the video

| Ground Truth Label | AI Description |
|---|---|
| "front crawl" | "A swimmer performing backstroke in a pool" |
| "white flowers blooming" | "Cherry blossoms on a tree against a blue sky" |

Both are displayed as **overlays** in FiftyOne for instant visual comparison

---

## How It Works

```
1. Select video samples in FiftyOne App
2. Run "Check Video Label" operator
3. Each sample gets:
   → similarity_score  (float)
   → label_check       (MATCH / MISMATCH)
   → is_mislabeled     (true / false)
   → video_description  (AI-generated)
4. Filter & sort to find the worst offenders
```

**No model training. No fine-tuning. No waiting.**

---

## Demo

![w:900](output_edited.gif)

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              FiftyOne App                    │
│  Select samples → Run operator → See results│
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   LabelGuard Plugin │
        └──────────┬──────────┘
                   │
     ┌─────────────▼─────────────┐
     │      Twelve Labs APIs      │
     │  Marengo Embed + Analyze   │
     └───────────────────────────┘
```

---

## Output Fields

| Field | Type | Example |
|---|---|---|
| `similarity_score` | float | `0.28` |
| `label_check` | string | `MISMATCH` |
| `is_mislabeled` | boolean | `True` |
| `video_description` | Classification | "A swimmer doing backstroke" |

Filter in FiftyOne:
```python
dataset.match(F("is_mislabeled") == True)
dataset.sort_by("similarity_score")
```

---

## Why Marengo?

- **Joint video-text embedding space** — no separate models needed
- **Zero-shot** — works on any label vocabulary out of the box
- **No index required** — embed directly, no setup overhead
- **1024-dim vectors** — fast cosine similarity comparison
- **Analyze API** — human-readable descriptions for explainability

---

## Tech Stack

| Component | Role |
|---|---|
| **FiftyOne** | Dataset curation & visualization |
| **Twelve Labs Marengo** | Video-text embeddings |
| **Twelve Labs Analyze** | Video description generation |
| **NumPy** | Cosine similarity computation |

---

# Thank You

**LabelGuard** — because you shouldn't need to train a model to find out your training data is wrong

GitHub: **github.com/pavithralagisetty/labelguard**

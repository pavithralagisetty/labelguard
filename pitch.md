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

Twelve Labs Marengo embeds video and text into the **same 1024-dim vector space**

```
Video file  →  Marengo video embedding
Label text  →  Marengo text embedding
cosine_similarity(video, text)  →  MATCH or MISMATCH
```

Plus Twelve Labs Analyze generates an **AI description** of the video — so you see *why* the label is wrong, not just that it is

---

## Demo

![w:900](output_edited.gif)

---

## How It Works

1. **Select** video samples in the FiftyOne App
2. **Run** the "Check Video Label" operator
3. **Get results** on every sample:

| Field | What It Tells You |
|---|---|
| `similarity_score` | How well the label matches the video (0 to 1) |
| `label_check` | `MATCH` or `MISMATCH` |
| `is_mislabeled` | `true` / `false` — filterable in one click |
| `video_description` | AI-generated description overlaid on the sample |

**No model training. No fine-tuning. No waiting.**

---

## Tech Stack

| Component | Role |
|---|---|
| **FiftyOne** | Dataset curation & visualization |
| **Twelve Labs Marengo Embed** | Joint video-text embeddings |
| **Twelve Labs Analyze** | Video description generation |
| **NumPy** | Cosine similarity computation |

- **Zero-shot** — works on any label vocabulary out of the box
- **No index required** — embed directly, no setup overhead
- **Explainable** — AI descriptions show what the model actually sees

---

# Thank You

**LabelGuard** — because you shouldn't need to train a model
to find out your training data is wrong

GitHub: **github.com/pavithralagisetty/labelguard**

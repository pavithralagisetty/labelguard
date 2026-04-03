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

---

# Thank You

**LabelGuard** — because you shouldn't need to train a model
to find out your training data is wrong

GitHub: **github.com/pavithralagisetty/labelguard**

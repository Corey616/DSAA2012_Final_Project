# Multi-Frame Story Generation System - Presentation

## DSAA2012 Final Project - Task 1: Story

---

## How to Use This Presentation

### Open the Presentation
1. Open `index.html` in any modern web browser (Chrome, Firefox, Safari, Edge)
2. Use arrow keys or click to navigate between slides

### Navigation Controls
- **Right Arrow / Space / Click Right Half**: Next slide
- **Left Arrow / Click Left Half**: Previous slide
- **Export PDF Button**: Print/save as PDF

### Slide Structure (10 slides, ~5 minutes)

| Slide | Title | Duration |
|-------|-------|----------|
| 1 | Title Page | 15s |
| 2 | Problem Definition & Challenges | 45s |
| 3 | System Architecture | 30s |
| 4 | Core Technical Innovations | 30s |
| 5 | Key Case Fixes & Results | 45s |
| 6 | Generated Storyboard Examples | 30s |
| 7 | Quantitative Results | 30s |
| 8 | Innovation Contributions | 30s |
| 9 | Project Structure & Statistics | 30s |
| 10 | Summary & Future Work | 30s |

---

## Image Gallery

Available images in `/images/` folder:

### Storyboards (Complete 3-frame sequences)
- `case01_storyboard.png` - Lily Kitchen (CLIP: 0.310, LPIPS: 0.393)
- `case02_storyboard.png` - Ryan Bus (Best overall: 0.357)
- `case03_storyboard.png` - Two characters
- `case04_storyboard.png` - Milo Toys (Toddler deduplication fix)
- `case06_storyboard.png` - Jack & Sara (Multi-character fix)
- `case07_storyboard.png` - Two characters
- `case09_storyboard.png` - Single character
- `case11_storyboard.png` - Best CLIP: 0.366
- `case17_storyboard.png` - Best LPIPS: 0.444
- `extra01_storyboard.png` - Extra test case
- `extra05_storyboard.png` - Extra test case

### Individual Frames
- `case01_frame1.png`, `case01_frame2.png`, `case01_frame3.png` - Lily frames
- `case04_frame1.png`, `case04_frame2.png`, `case04_frame3.png` - Milo frames
- `case06_frame1.png`, `case06_frame2.png`, `case06_frame3.png` - Jack & Sara frames

---

## Key Results Summary

### Performance Metrics
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Avg CLIP Score | >=0.25 | **0.288** | PASS |
| Avg LPIPS | >=0.30 | **0.386** | PASS |
| Per Frame Time | <15s | **14.3s** | PASS |
| VRAM Peak | <20GB | **18GB** | PASS |
| OOM Rate | <5% | **0%** | PASS |

### Test Coverage
- **Standard Tests**: 20 stories (01-20)
- **Extra Tests**: 12 stories (extra_01-12)
- **Total**: 32 stories, 96 frames, **100% success rate**

---

## Scoring Criteria Mapping

| Criterion | Slides | Key Content |
|-----------|--------|-------------|
| Technical Soundness (25%) | 2, 3, 4 | Problem definition, architecture, innovations |
| Novelty and Performance (25%) | 4, 7, 8 | Dynamic weights, deduplication, quantization |
| Completeness and Rigor (25%) | 5, 6, 7 | Case fixes, examples, quantitative results |
| Presentation Clarity (25%) | All | Structure, visuals, flow |

---

## Technical Dependencies

The presentation uses:
- **Mermaid.js** (loaded from CDN) for architecture diagrams
- No other external dependencies

---

## Project Information

- **Course**: DSAA2012 Final Project
- **Task**: Task 1: Story - Multi-frame story generation
- **Models**: SDXL Base 1.0 + Qwen2.5-7B-Instruct (local)
- **Output**: 1024x1024 PNG images
- **Batch Test**: `python run_taska_batch.py --gpu <id>`

---

Generated: 2026-04-27

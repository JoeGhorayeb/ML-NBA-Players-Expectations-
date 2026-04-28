# ML-NBA-Players-Expectations-
A machine learning project for my ML workshop that predicts whether an NBA draft pick met, exceeded, or underperformed expectations. It factors in draft position, team prestige, physical profile, and performance stats to output an expectation score and classification label through a Flask web app.

# 🏀 NBA Draft Expectation Predictor

A machine learning project that predicts whether an NBA draft pick met, exceeded, or underperformed the expectations set on them — based on their physical profile, draft position, team context, and on-court performance stats.

---

## Overview

Not all draft picks are created equal. A #1 pick on an elite franchise carries far more pressure than a late second-rounder on a rebuilding team. This project quantifies that pressure mathematically and uses it to evaluate whether a player truly lived up to what was expected of them.

The model outputs two predictions:
- **Expectation Score** — a continuous percentage reflecting how much of their expectations a player fulfilled
- **Classification** — whether the player `Exceeded`, `Met`, or `Underperformed` expectations

---

## How It Works

Rather than using raw stats alone, the pipeline engineers a set of context-aware features:

- **BMI Similarity** — how close a player's body composition is to the ideal NBA profile
- **Draft Expectation** — pressure derived from draft position using an inverse square function
- **Team Strength** — the historical prestige of the drafting franchise
- **Pressure Factor** — a combined metric of team and draft pressure
- **Adjusted Stats** — per-game stats scaled by the pressure context the player performed under
- **Performance Score (S)** — a sigmoid-normalized composite of all adjusted stats
- **BPM Score (sb)** — impact rating normalized across the league

These features feed into two separate models:

| Task | Model | Score |
|------|-------|-------|
| Regression | Ridge | R² 0.86 |
| Classification | Random Forest | Macro F1 0.74 |

SMOTE oversampling is applied to handle the natural class imbalance — most draft picks realistically underperform their expectations.

---

## Features Used

`height_cm` · `weight_kg` · `draft_pick` · `team` · `ppg` · `rpg` · `apg` · `spg` · `bpg` · `BPM`

---

## Web App

Built with **Flask + HTML/CSS** — input a player's raw stats and get an instant prediction with a color-coded result card.

- 🟢 **Exceeded** — player significantly surpassed expectations
- 🔵 **Met** — player performed in line with expectations  
- 🔴 **Underperformed** — player fell short of expectations

---

## Stack

- Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn
- Matplotlib · Seaborn
- Flask · HTML · CSS

---

## Dataset

550 NBA draft picks with career averages, physical measurements, draft position, and team context.

---

## Limitations

- Team rankings are static and do not reflect year-by-year franchise strength
- minority classes (Met / Exceeded) are naturally small — classification improves with more data
- Model is trained on historical data and may not generalize to modern playstyles

---

*Built as an end-to-end ML project — from raw data and feature engineering to a deployed web interface.*

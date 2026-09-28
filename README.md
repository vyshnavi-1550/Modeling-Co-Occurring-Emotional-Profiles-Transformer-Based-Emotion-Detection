# Modeling Co-Occurring Emotional Profiles: Transformer-Based Emotion Detection

## 📌 Overview

This project analyzes how emotions co-occur in text and uses their relationships to identify higher-level emotional profiles.

The project uses Google's **GoEmotions** dataset containing 28 emotion categories. Emotion co-occurrence patterns are represented as a graph, and **Louvain community detection** is used to identify groups of related emotions.

The discovered emotions are mapped into three higher-level emotional profiles:

- **Epistemic Profile**
- **Optimistic and Positive Profile**
- **Frustration and Negative Profile**

Machine learning and Transformer-based models are then trained to classify text into these emotional profiles.

The project also includes a **Gradio GUI** for interactive prediction.

---

## 🔄 Project Pipeline

```text
GoEmotions Dataset
        ↓
Emotion Label Processing
        ↓
Binary Emotion Matrix
        ↓
Emotion Co-occurrence Matrix
        ↓
Emotion Graph
        ↓
Louvain Community Detection
        ↓
Emotional Communities
        ↓
Three Emotional Profiles
        ↓
Supervised Classification Dataset
        ↓
SVM / BERT / RoBERTa
        ↓
Model Evaluation
        ↓
Gradio GUI

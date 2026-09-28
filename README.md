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
```

---

## 📊 Dataset

The project uses Google's **GoEmotions** dataset.

### Dataset Details

| Item | Value |
|---|---|
| **Dataset** | GoEmotions |
| **Source** | Google Research |
| **Text Source** | Reddit comments |
| **Emotion Categories** | 28 |
| **Annotation Type** | Multi-label |
| **Final Classification Task** | 3 emotional profiles |

The 28 emotion categories include:

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

> The dataset is **not stored in this GitHub repository**. It is downloaded using the Hugging Face `datasets` library.

---

## 🧠 Methodology

### 1. Emotion Co-occurrence Analysis

The multi-label emotion annotations are converted into a binary emotion matrix.

- `1` → Emotion is present
- `0` → Emotion is absent

A co-occurrence matrix is created to measure how frequently pairs of emotions occur together.

### 2. Emotion Graph Construction

An emotion graph is constructed using **NetworkX**.

- **Nodes** represent emotions.
- **Edges** represent emotion co-occurrence.
- **Edge weights** represent co-occurrence frequency.

A threshold is applied to remove weak relationships.

### 3. Louvain Community Detection

The **Louvain community detection** algorithm is applied to the emotion graph to identify groups of strongly connected emotions.

### 4. Emotional Profile Construction

The detected emotion communities are interpreted and mapped into three higher-level emotional profiles.

### 5. Supervised Classification

The derived emotional profiles are used as labels to create a supervised text classification dataset.

The following models are trained and compared:

- SVM
- BERT
- RoBERTa

---

## 😊 Emotional Profiles

The project uses three final emotional profiles:

| Profile | Description |
|---|---|
| **Epistemic Profile** | Emotions related to curiosity, confusion, realization and surprise |
| **Optimistic and Positive Profile** | Positive and optimistic emotional patterns |
| **Frustration and Negative Profile** | Negative emotional patterns including frustration, anger, sadness and related emotions |

---

## 🤖 Models

### SVM

A **Support Vector Machine (SVM)** is used as a classical machine-learning baseline.

### BERT

The project fine-tunes:

```text
bert-base-uncased
```

### RoBERTa

The project fine-tunes:

```text
roberta-base
```

---

## 📈 Evaluation

Models are compared using accuracy, precision, recall, and macro F1-score.

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM | TBD | TBD |
| BERT | TBD | TBD |
| RoBERTa | TBD | TBD |

---

## 🖥️ Gradio GUI

An interactive Gradio interface lets you enter text and see the predicted emotional profile.

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

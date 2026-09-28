# Modeling Co-Occurring Emotional Profiles: Transformer-Based Emotion Detection

> A graph-based emotion profiling system using Louvain community detection and Transformer-based classification.

## 📌 Overview

This project analyzes how emotions co-occur in text and uses their relationships to identify higher-level emotional profiles.

It uses Google's **GoEmotions** dataset, which contains 28 emotion categories. Emotion co-occurrence patterns are represented as a graph, and **Louvain community detection** is used to identify groups of related emotions.

The discovered emotions are mapped into three higher-level emotional profiles:

- **Epistemic Profile**
- **Optimistic and Positive Profile**
- **Frustration and Negative Profile**

SVM, BERT and RoBERTa models are then trained to classify text into these profiles. The project also includes a **Gradio GUI** for interactive prediction.

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

| Item | Value |
|---|---|
| Dataset | GoEmotions |
| Source | Google Research |
| Text Source | Reddit comments |
| Samples used | 43,410 |
| Emotion Categories | 28 |
| Annotation Type | Multi-label |
| Final Classification Task | 3 emotional profiles |

The 28 emotion categories are:

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

> The dataset is **not stored in this repository**. It is downloaded using the Hugging Face `datasets` library.

---

## 🧠 Methodology

### 1. Emotion Co-occurrence Analysis

The multi-label annotations are converted into a binary emotion matrix (`1` = emotion present, `0` = absent). A co-occurrence matrix then measures how often pairs of emotions appear together in the same text.

### 2. Emotion Graph Construction

An emotion graph is built using **NetworkX**:

- **Nodes** represent emotions.
- **Edges** represent emotion co-occurrence.
- **Edge weights** represent co-occurrence frequency.

A threshold removes weak relationships. The resulting network has **28 nodes and 47 edges**.

### 3. Louvain Community Detection

The **Louvain algorithm** is applied to the emotion graph to identify groups of strongly connected emotions.

### 4. Emotional Profile Construction

The detected communities are interpreted and mapped into three higher-level emotional profiles.

### 5. Supervised Classification

The derived profiles are used as labels to create a supervised text classification dataset. SVM, BERT and RoBERTa are trained and compared.

---

## 😊 Emotional Profiles

| Profile | Description |
|---|---|
| **Epistemic Profile** | Emotions related to curiosity, confusion, realization and surprise |
| **Optimistic and Positive Profile** | Positive and optimistic emotional patterns |
| **Frustration and Negative Profile** | Negative emotional patterns including frustration, anger, sadness and related emotions |

---

## 🤖 Models

### SVM

A **Support Vector Machine** is used as a classical machine-learning baseline.

### BERT

The project fine-tunes:

```text
bert-base-uncased
```

for three-class emotional profile classification.

### RoBERTa

The project fine-tunes:

```text
roberta-base
```

for three-class emotional profile classification.

BERT and RoBERTa are trained using PyTorch and Hugging Face Transformers.

---

## ⚙️ Experimental Setup

All models use the same fixed train-test split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

| Setting | Value |
|---|---|
| Training data | 80% |
| Testing data | 20% |
| Random state | 42 |
| Stratification | Yes |

---

## 📈 Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM | 82.38% | 70.04% |
| BERT | 94.85% | 93.00% |
| RoBERTa | 95.25% | 93.41% |

Models are evaluated using accuracy, precision, recall, macro F1-score, weighted F1-score, Hamming loss, Jaccard score, confusion matrix and classification report.

---

## 🖥️ Gradio GUI

The project includes a Gradio-based GUI for interactive emotional profile prediction. Enter a text and the app returns the predicted profile.

**Example**

Input:

```text
Thanks! I love watching him every week.
```

Output:

```text
Optimistic and Positive Profile
```

**GUI Flow**

```text
User Input → Text Tokenization → Trained Transformer Model → Profile Prediction
```

---

## 🛠️ Technologies Used

Python, Pandas, NumPy, Scikit-learn, NetworkX, python-louvain, SciPy, PyTorch, Hugging Face Transformers, Hugging Face Datasets, BERT, RoBERTa, Matplotlib, Seaborn, Gradio

---

## 💻 Installation

```bash
pip install numpy pandas networkx matplotlib seaborn datasets scipy scikit-learn python-louvain torch transformers tqdm gradio
```

---

## 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/vyshnavi-1550/Modeling-Co-Occurring-Emotional-Profiles-Transformer-Based-Emotion-Detection.git
```

2. Open `emotion_profiling.ipynb` in Jupyter Notebook, JupyterLab or Google Colab.

3. Run the notebook from top to bottom. It performs dataset loading, co-occurrence analysis, graph construction, Louvain community detection, profile construction, SVM / BERT / RoBERTa training, evaluation and the Gradio GUI.

A GPU is recommended for BERT and RoBERTa training.

---

## 📁 Project Structure

```text
Modeling-Co-Occurring-Emotional-Profiles-Transformer-Based-Emotion-Detection/
│
├── README.md
└── emotion_profiling.ipynb
```

---

## 🔮 Future Improvements

- Test additional Transformer models.
- Explore more detailed emotional profiles.
- Add prediction confidence scores to the GUI.
- Explore multilingual emotion classification.
- Deploy the Gradio application online.
- Evaluate the approach on additional emotion datasets.

---

## 📚 References

- Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL 2020.
- GoEmotions on GitHub: https://github.com/google-research/google-research/tree/master/goemotions
- GoEmotions on Hugging Face: https://huggingface.co/datasets/google-research-datasets/go_emotions
- Blondel, V. D., et al. (2008). *Fast unfolding of communities in large networks*.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.
- Hugging Face Datasets, Hugging Face Transformers, NetworkX, Gradio.

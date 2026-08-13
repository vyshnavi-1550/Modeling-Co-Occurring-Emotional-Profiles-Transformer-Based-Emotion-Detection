# Modeling Co-Occurring Emotional Profiles: Transformer-Based Emotion Detection

This project studies how emotions **co-occur** in text and uses that structure to define higher-level **emotional profiles**. Starting from the 28 fine-grained emotion labels in the [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset, it builds a co-occurrence graph, clusters it into interpretable profiles (e.g., *Frustration and Negative*, *Optimistic and Positive*, *Epistemic*), and then trains and compares classical ML and **transformer-based models (BERT, RoBERTa)** to detect these profiles directly from raw text.

## Overview

GoEmotions labels Reddit comments with one or more of 28 emotions (plus neutral). Because a single comment can carry multiple emotions, this project treats emotion co-occurrence as a signal for discovering broader emotional "profiles" (e.g., emotions like anger, disgust, and sadness tend to co-occur and form a "Frustration and Negative" profile).

The pipeline:

1. **Load & vectorize** the GoEmotions dataset into a binary emotion matrix (rows = texts, columns = 28 emotions).
2. **Build a co-occurrence matrix** capturing how often emotion pairs appear together in the same text.
3. **Construct a graph** where nodes are emotions and edges are weighted by co-occurrence strength (above a threshold).
4. **Cluster the graph** using Louvain community detection (and cross-check with hierarchical/agglomerative clustering) to discover emotion communities.
5. **Name and validate the clusters** as human-interpretable emotional profiles, e.g.:
   - Optimistic and Positive Profile
   - Frustration and Negative Profile
   - Epistemic Profile (curiosity, confusion, realization, surprise)
   - Affiliative/Neutral, Fear, Nervousness, Embarrassment, Grief, Pride, Relief
6. **Simplify to a 3-profile scheme** (Epistemic, Optimistic/Positive, Frustration/Negative) for supervised learning.
7. **Build a supervised dataset** mapping each text to its derived profile label.
8. **Train and compare classifiers** to predict the profile from raw text / emotion features:
   - Logistic Regression (TF-IDF and one-hot emotion features)
   - Linear SVM
   - BERT (`bert-base-uncased`)
   - RoBERTa (`roberta-base`)
9. **Evaluate models** using accuracy, macro/weighted F1, Hamming loss, and Jaccard score, and inspect model weights to identify the top words driving each profile.
10. **Interactive demo** — a Gradio GUI that takes free-text input and predicts its emotional profile using the trained model.

## Dataset

- **Source:** `google-research-datasets/go_emotions` (via Hugging Face `datasets`)
- **Labels:** 27 emotions + neutral, multi-label per text
- Several intermediate CSVs are produced along the way:
  - `emotion_cluster_dataset.csv` — text + raw cluster ID
  - `filtered_cluster_dataset.csv` — cluster dataset with minor/noisy clusters removed
  - `identified_profiles.csv` / `identified_profiles_table.csv` — cluster-to-profile name mapping and summaries
  - `supervised_dataset.csv` — final text + profile label dataset used for model training

## Methods

- **Graph construction:** `networkx`, spring layout for visualization
- **Community detection:** `python-louvain` (`community_louvain`) for modularity-based clustering
- **Hierarchical clustering:** `scipy.cluster.hierarchy` (Ward linkage) as a cross-check, visualized as a dendrogram
- **Dimensionality/pattern exploration:** `sklearn.decomposition.NMF`
- **Classical ML:** `TfidfVectorizer` + `LogisticRegression` / `LinearSVC` (scikit-learn)
- **Deep learning:** `transformers` (BERT, RoBERTa) fine-tuned for 3-class sequence classification, trained with PyTorch
- **Evaluation:** accuracy, macro/weighted F1, Hamming loss, Jaccard score, confusion matrices, classification reports
- **Explainability:** logistic regression weight-matrix heatmaps and top positive/negative words per profile
- **Demo UI:** `gradio` app that loads the trained classifier and returns the predicted emotional profile for arbitrary input text

## Requirements

```
numpy
pandas
networkx
matplotlib
seaborn
datasets
scipy
scikit-learn
python-louvain
torch
transformers
tqdm
gradio
```

Install with:

```bash
pip install numpy pandas networkx matplotlib seaborn datasets scipy scikit-learn python-louvain torch transformers tqdm gradio
```

## Usage

1. Open and run `emotion_profiling.ipynb` top to bottom in a Jupyter environment (a GPU is recommended for the BERT/RoBERTa training cells).
2. The notebook will:
   - Download GoEmotions automatically via `datasets`
   - Build the co-occurrence graph and cluster it into emotional profiles
   - Export intermediate CSVs (listed above) into the working directory
   - Train and evaluate the classical and transformer-based models
3. Run the final **GUI** cell to launch a local Gradio app where you can type in text and get a predicted emotional profile (Optimistic & Positive / Frustration & Negative / Epistemic).

## Project Structure

Since this project currently lives entirely inside a single notebook, the logical structure is:

| Section | Description |
|---|---|
| Data Loading | Load GoEmotions, build binary emotion matrix |
| Co-occurrence & Graph | Build co-occurrence matrix and emotion graph |
| Clustering | Louvain community detection + hierarchical clustering, cluster naming |
| Profile Construction | Map clusters/emotions to a 3-profile scheme (Epistemic / Optimistic / Frustration) |
| Supervised Dataset | Build text → profile-label dataset, export CSVs |
| Modeling | Logistic Regression, SVM, BERT, RoBERTa training and evaluation |
| Evaluation | Accuracy, F1, Hamming loss, Jaccard score, weight/word importance analysis |
| GUI | Gradio app for interactive profile prediction |

## Notes

- The notebook contains iterative/exploratory work (multiple redefinitions of the same variables, alternate cluster-naming schemes, etc.) as part of the research process; the final supervised task settles on **3 profiles**: Epistemic, Optimistic and Positive, and Frustration and Negative.
- Cluster names and thresholds (e.g., the co-occurrence edge threshold of 50) were chosen empirically and can be tuned for different granularities of emotional profiles.

## License

Add a license of your choice here. The GoEmotions dataset itself is released by Google Research under its own license — check the [dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions) for terms of use.

=====================================================
DRONE DETECTION PROJECT: README & PIPELINE OVERVIEW
=====================================================


## Objective
To compare the performance of YOLO (You Only Look Once) and Faster R-CNN for drone detection at varying distances, leveraging a combination of real-world and simulated (virtual) data.

## Directory Structure

/
├── drone_detection_distance/
│   ├── scripts/
│   │   └── deterministic_var_nobbox.py  <-- Main file for data preparation 
│   ├── notebooks/
│   │   ├── yolo_python.ipynb            <-- YOLO training/evaluation pipeline
│   │   ├── fast_rnn_tse.ipynb           <-- Faster R-CNN training/t-SNE pipeline
│   │   └── runs_yolo/                   <-- **Output folder for all YOLO statistics (mAP, loss, etc.)**
│   └── tsne_plots/                    <-- **Output folder for t-SNE visualizations**
├── real.yaml                          <-- Data config: 100% Real Data
├── virtual.yaml                       <-- Data config: 100% Virtual Data
├── mixed.yaml                         <-- Data config: Standard Mix (e.g., 50/50)
├── mixed_70_30.yaml                   <-- Data config: Specific 70% Real / 30% Virtual
└── README.md                          <-- (This is the file being built)


## Project Pipeline Overview

The pipeline is executed in three main sequential phases: Data Configuration, Model Training & Evaluation, and Feature Analysis.

### 1. Data Configuration (YAML Files)
These files define the dataset splits used for training, allowing researchers to quickly test different domain adaptation scenarios (real vs. virtual data ratios).

*   **`real.yaml`**: Used for baseline evaluation on only real-world data.
*   **`virtual.yaml`**: Used for baseline evaluation on only simulated data.
*   **`mixed.yaml` / `mixed_70_30.yaml`**: Used for testing model robustness and transfer learning performance on mixed datasets.

### 2. Model Training & Evaluation (Notebooks)

| Step | File/Directory | Description | Output Location |
|---|---|---|---|
| **Data Generation** | `drone_detection_distance/scripts/deterministic_var_nobbox.py` | **MANDATORY FIRST STEP.** This script creates the necessary dataset and file structures required for training the models. | N/A (creates input data) |
| **YOLO Pipeline** | `drone_detection_distance/notebooks/yolo_python.ipynb` | Loads a specified YAML configuration, trains the **YOLO** model, and evaluates its performance, focusing on metrics like mAP and distance-based detection rates. | `drone_detection_distance/notebooks/runs_yolo/` |
| **Faster R-CNN Pipeline**| `drone_detection_distance/notebooks/fast_rnn_tse.ipynb` | Loads a specified YAML configuration, trains the **Faster R-CNN** model, evaluates performance, and proceeds to feature analysis. | N/A (model checkpoints stored separately) |

### 3. Feature Analysis (t-SNE)

*   **Analysis Output**: `drone_detection_distance/tsne_plots/`
*   **Execution**: Integrated within the `fast_rnn_tse.ipynb` notebook.
*   **Purpose**: Generates plots using **t-SNE** (t-distributed Stochastic Neighbor Embedding) to visualize high-dimensional feature vectors. This helps in understanding how effectively the Faster R-CNN model separates features originating from real data versus those from virtual/simulated data.
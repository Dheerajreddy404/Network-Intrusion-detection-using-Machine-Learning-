# Machine Learning Based Network Intrusion Detection System

## Overview

This project implements a Machine Learning-based Network Intrusion Detection System (NIDS) using flow-based network traffic features. The system was developed as part of a Bachelor Thesis project focused on evaluating intrusion detection performance using both a self-generated dataset and the UNSW-NB15 benchmark dataset.

The project uses Random Forest and XGBoost models to classify network traffic into normal and malicious categories.

---

## Objectives

* Generate a self-created intrusion detection dataset
* Extract flow-based statistical features from network traffic
* Train and evaluate machine learning models
* Compare Random Forest and XGBoost performance
* Analyze the impact of dataset imbalance on intrusion detection

---

## Attack Types Used

The following traffic categories were included in the experiments:

* Normal Traffic
* SSH Brute Force
* DoS Attacks
* Port Scanning

---

## Experimental Environment

* Kali Linux (Attack Machine)
* Ubuntu Linux (Target Machine)
* Wireshark for packet capture
* Python for preprocessing and model training

---

## Workflow

1. Generate network traffic
2. Capture packets using Wireshark
3. Export traffic to CSV
4. Generate flow-based records
5. Extract statistical features
6. Preprocess dataset
7. Train Random Forest and XGBoost models
8. Evaluate model performance

---

## Extracted Features

* packet_count
* total_bytes
* avg_packet_size
* duration

---

## Machine Learning Models

### Random Forest

* Ensemble learning using bagging
* Robust against imbalance and overfitting

### XGBoost

* Gradient boosting based classifier
* High predictive performance

---

## Results

### Self-Generated Dataset

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 91.8%    |
| XGBoost       | 90.1%    |

### UNSW-NB15 Benchmark Dataset

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 93.1%    |

---

## Key Findings

* Random Forest achieved more stable class-wise performance
* XGBoost struggled with minority-class PortScan detection
* Dataset imbalance strongly affected model behavior
* Flow-based traffic features were effective for intrusion detection

---

## Technologies Used

* Python
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib
* Wireshark

---

## Future Improvements

* Real-time intrusion detection
* Deep learning approaches
* Better dataset balancing using SMOTE
* Additional attack categories
* Larger-scale traffic generation

---

## Thesis

Bachelor Thesis:
Machine Learning Based Network Intrusion Detection System

---

## License

This project is developed for academic and research purposes.

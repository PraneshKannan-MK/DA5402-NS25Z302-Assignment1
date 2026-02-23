# DA5402 Manual MLOps Assignment — Predictive Maintenance System

## Overview
This project demonstrates a Manual MLOps pipeline for a Predictive Maintenance system using the AI4I 2020 dataset.  
The goal is to manage the full lifecycle — data versioning, model training, deployment, and monitoring — without automated MLOps tools.

---

## Project Structure

manual_mlops_project/
│
├── data/
├── models/
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── inference.py
│   ├── monitor.py
│   ├── test_api.py
│   └── utils.py
│
├── config.yaml
├── requirements.txt
├── deployment_log.csv

---

## Setup Instructions

### Activate Environment
python -m venv .venv
.\.venv\Scripts\Activate

### Install Dependencies
pip install -r requirements.txt

---

## Execution Flow

### Data Preparation
python -m src.data_prep

Creates processed dataset and updates manifest.txt.

### Model Training
python -m src.train

Trains model and updates manual model registry.

### Start API
uvicorn src.inference:app --reload --port 5050

Starts FastAPI deployment and logs deployment history.

### Smoke Tests
python src/test_api.py

Validates API functionality.

### Monitoring
python -m src.monitor

Calculates production error rate and compares with training baseline.

---

## Screencast
A screencast demonstrating the working pipeline, explanation of approach, and challenges encountered is included with this submission.

---

## Documentation Requirement
This repository includes a clear and structured guide explaining how to understand, set up, and run the project.  
The workflow and commands are documented to ensure reproducibility during evaluation.

---

## Manual MLOps Components
- Data Versioning via manifest.txt
- Model Registry via model_metadata.log
- Deployment Tracking via deployment_log.csv
- Monitoring & Drift Detection via monitor.py

---

## Constraints
No automated MLOps tools (MLflow, DVC, Airflow, Kubernetes) were used.

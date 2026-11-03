# 🏥 StepWise PRO: Multimodal AI Decision Support Engine for Neurodegenerative Triage

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.0-black.svg)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai/)
[![HL7 FHIR R4](https://img.shields.io/badge/HL7-FHIR%20R4-orange.svg)](https://hl7.org/fhir/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **StepWise PRO** is an enterprise-grade, multimodal Clinical Decision Support System (CDSS) for early Alzheimer’s disease detection, 24-month rapid progression velocity forecasting, and Disease-Modifying Therapy (DMT) eligibility authorization.

---

## 📑 Table of Contents
- [Clinical Motivation & 4-Stage Gated Funnel](#-clinical-motivation--4-stage-gated-funnel)
- [System Architecture](#-system-architecture)
- [AI/ML Modeling & Multi-Task Vision Heads](#-aiml-modeling--multi-task-vision-heads)
- [Benchmark Performance Summary](#-benchmark-performance-summary)
- [Demographic Fairness (LASI-DAD Engine)](#-demographic-fairness-lasi-dad-engine)
- [HL7 FHIR R4 & CDS Hooks Integration](#-hl7-fhir-r4--cds-hooks-integration)
- [Repository Structure & Detailed Docs](#-repository-structure--detailed-docs)
- [Quickstart Guide](#-quickstart-guide)
- [Author & Citation](#-author--citation)

---

## 🔬 Clinical Motivation & 4-Stage Gated Funnel

Conventional medical AI models suffer from two critical flaws:
1. **Static Classification:** Guessing a patient's current symptoms rather than predicting future trajectory.
2. **Missing Modality Failure:** End-to-end models requiring an expensive $3,200 PET scan or MRI before giving any prediction, making them useless in primary care.

**StepWise PRO solves this by predicting 24-Month Rapid Progression Velocity ($R_{24m} \in [0, 1]$)** through a stepped, cost-effective diagnostic cascade:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STEPWISE 4-STAGE GATED WORKFLOW                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [Stage 1: Primary Care / PHC]  ──▶ Non-Invasive Intake (MMSE, MoCA, CDR-SB, FAQ, Vitals)
                                      │ (Escalation Gate 1: R1 >= 0.0737)
                                      ▼
 [Stage 2: Phlebotomy Lab]      ──▶ Single Blood Draw: 7-Assay Harmonized p-tau217/Aβ + APOE-ε4
                                      │ (Escalation Gate 2: R2 >= 0.0978)
                                      ▼
 [Stage 3: Radiology Suite]     ──▶ 3D Volumetric MRI Morphometry (Hippocampal Atrophy, WMH)
                                      │ (Escalation Gate 3: R3 >= 0.1330)
                                      ▼
 [Stage 4: Molecular PET]       ──▶ Amyloid PET Centiloids + Tau SUVR + ARIA Safety Profiler
                                      │
                                      ▼
                        [1-Click FHIR DMT Order Authorization]
```

---

## 🏛️ System Architecture

```
                                  STEPWISE PRO FULL-STACK TOPOLOGY
                                  
   CLINICIAN FRONTEND               ENTERPRISE CDS BACKEND               AI / ML INFERENCE ENGINES
 ┌──────────────────────┐         ┌────────────────────────┐         ┌───────────────────────────────┐
 │ Next.js 14 Dashboard │ ◄─────► │ FastAPI Core Engine    │ ◄─────► │ 4x Calibrated XGBoost Models  │
 │ • 4-Stage Stepper    │ (REST)  │ • Native NaN Router    │         │ • Isotonic Platt Calibrator   │
 │ • Voice Dictation    │         │ • LASI-DAD Transformer │         │ • TreeSHAP Attribution        │
 │ • Dual DICOM Viewer  │         │ • SQLite Longitudinal  │         ├───────────────────────────────┤
 │ • FHIR Order Modal   │         │ • Lab PDF OCR Parser   │ ◄─────► │ 2x Multi-Task ResNet-18 Heads │
 └──────────────────────┘         └────────────────────────┘         │ • MRI Morphometry (HVR)       │
                                              │                      │ • PET Centiloids (0-100+ CL)  │
                                              ▼                      └───────────────────────────────┘
                                   HL7 FHIR R4 / CDS HOOKS
```

---

## 🧠 AI/ML Modeling & Multi-Task Vision Heads

### 1. Tabular Staged Progression XGBoost
- Trained on **9,453 longitudinal clinical visits across 2,458 patients** from USC LONI ADNI.
- Enforces **monotonic constraints** ($\uparrow p\text{-tau217} \implies \text{Risk } +1$, $\downarrow \text{MoCA} \implies \text{Risk } -1$) to guarantee FDA/CE-MDR auditability.
- Native missing value (`NaN`) branching handles missing clinical features without imputation artifacts.
- Out-of-Fold (OOF) Isotonic probability calibration minimizes Brier Score loss ($<0.05$).

### 2. Multi-Task Deep Learning Vision Heads
- **Stage 3 MRI Vision Head (`ResNet-18`):** Ingests 2.5D tri-planar structural slices and 5 physical priors, refining bilateral Hippocampal Volume ($cm^3$), Ventricles ($cm^3$), and Hippocampal-to-Ventricle Ratio ($\text{HVR}$ MAE = $0.0024$).
- **Stage 4 Molecular PET Centiloid Head (`ResNet-18`):** Direct quantification of Amyloid Centiloid burden ($0-100+\text{ CL}$, MAE = $0.31\text{ CL}$, $R^2 = 0.999$) and $A^+/A^-$ biological classification.

---

## 📊 Benchmark Performance Summary

| Stage & Modality | Model Architecture | Validation Performance | Key Clinical Metric / Threshold |
| :--- | :--- | :--- | :--- |
| **Stage 1: Cognitive/EHR** | XGBoost + Platt Calibrator | **0.808 CV / 0.717 Holdout AUC** | $T_1 = 0.0737$ (Sens: 51.4%, Spec: 83.3%) |
| **Stage 2: Blood Biofluids** | 7-Assay Z-Score XGBoost | **0.815 CV / 0.696 Holdout AUC** | Top 10% queue captures 44.1% progressors |
| **Stage 3: 3D Volumetric MRI** | Multi-Task 2.5D ResNet-18 | **0.9985 Macro AUC / 96.6% Acc** | HVR MAE = $0.0024$ (Sub-millimeter volume agreement) |
| **Stage 4: Molecular PET** | Multi-Task 2.5D ResNet-18 | **0.9997 Macro AUC / 97.7% Acc** | Centiloids MAE = $0.31\text{ CL}$, $100\%$ Specificity |

---

## 🌍 Demographic Fairness (LASI-DAD Engine)

To prevent misdiagnosis in populations with low formal education where standard cognitive exams (MMSE/MoCA) yield false positives:
$$\text{MMSE}_{\text{adjusted}} = \min\left(30, \text{MMSE}_{\text{raw}} + \max(0, 12 - \text{Education Years}) \times 0.35\right)$$
Combined with a $+20\%$ metabolic risk multiplier for White Matter Hyperintensities (WMH) associated with diabetic and hypertensive vascular disease.

---

## 🔌 HL7 FHIR R4 & CDS Hooks Integration

StepWise PRO natively formats recommendations into standardized HL7 FHIR R4 JSON payloads:
- **`ServiceRequest`:** 1-Click electronic authorization for 3.0T MRI, Molecular PET, or Lecanemab immunotherapy.
- **CDS Hooks (`patient-view`):** Emits actionable alert cards with automated clinical narratives directly into hospital EHR systems (Epic, Cerner).

---

## 📁 Repository Structure & Detailed Docs

```
StepWise/
├── backend/                   # FastAPI Enterprise CDS Service
│   ├── app.py                 # Core API endpoints & prediction cascades
│   ├── database.py            # SQLite patient registry & timeline tracking
│   ├── lab_parser.py          # PDF lab report OCR & biomarker extractor
│   ├── fhir_service.py        # HL7 FHIR R4 ServiceRequest generator
│   ├── roi_simulation.py      # Hospital capacity & economic ROI engine
│   └── models/                # Model feature schemas & metadata JSONs
├── frontend/                  # Next.js 14 Clinician Command Center
│   ├── src/app/page.tsx       # Interactive 4-stage clinician workstation
│   ├── src/app/components/    # Dual DICOM/PET viewer with Inferno heatmaps
│   └── src/components/        # TreeSHAP waterfall, Triage worklist, Lab modal
├── docs/                      # Comprehensive Technical Deep-Dives
│   ├── 01_CLINICAL_FOUNDATIONS_AND_ATN_FRAMEWORK.md
│   ├── 02_MULTIMODAL_DATA_STRATEGY_AND_COHORTS.md
│   ├── 03_AI_ML_MODELING_AND_EXPLAINABILITY.md
│   ├── 04_SYSTEM_ARCHITECTURE_AND_FHIR_DESIGN.md
│   └── 05_PROJECT_EXECUTION_AND_BENCHMARKS.md
├── evaluations/               # ROC curves, PR curves, Confusion Matrices
├── scripts/                   # Data preprocessing, harmonizers & training scripts
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Ashish-2028/StepWise.git
cd StepWise
```

### 2. Backend Setup (FastAPI)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Or install: fastapi uvicorn xgboost scikit-learn pydantic torch
uvicorn app:app --reload --port 8000
```
Swagger UI will be available at: `http://127.0.0.1:8000/docs`

### 3. Frontend Setup (Next.js 14)
```bash
cd ../frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 👤 Author & Maintainer

**Ashish Kumar**  
Lead Machine Learning & Systems Architect  
GitHub: [@Ashish-2028](https://github.com/Ashish-2028)  
Email: `ashishkumartiwari2028@gmail.com`

---
*Developed with clinical rigor, mathematical transparency, and demographic equity.*

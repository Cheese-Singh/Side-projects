# Side Projects Index

A collection of technical explorations and utility scripts.

## Contents

### [MLX-Series](./MLX-Series/)
* **Description:** Progressive exploration of the MLX framework for running and understanding machine learning models on Apple Silicon.
* **Focus:** From tensor operations and manual embeddings to local LLM inference using quantized models.
* **Files:**
  - `01_mlx_basics.py` → Tensor operations, vectorization, manual tokenization, and embedding layer implementation
  - `02_mlx_llm_inference.py` → Running a local Qwen3.5-9B model using MLX with configurable inference parameters
* **Stack:** `mlx`, `mlx-vlm`, `python`

---

### [research-trend-analyzer](./research-trend-analyzer/)
* **Description:** Automated tool for tracking scientific publication volume over time.
* **Focus:** REST API integration with OpenAlex, session handling, and data visualization.
* **Stack:** `requests`, `matplotlib`, `OpenAlex API`

---

### [catboost-loocv-regression](./catboost_loocv_regression.py)
* **Description:** CatBoost regression model to predict supercapacitor capacitance (F/g) from material and synthesis features.
* **Focus:** Handling small datasets with LOOCV, native categorical feature support, and diagnostic plot generation.
* **Stack:** `catboost`, `scikit-learn`, `pandas`, `matplotlib`

---

## Technical Objectives

The goal of this repository is to document:
- hands-on exploration of new frameworks (e.g., MLX)
- progression from low-level numerical operations to applied AI systems
- development of small, focused tools for data-driven workflows

Each module represents a step toward building more complex systems, with an emphasis on understanding underlying mechanics rather than only using high-level abstractions.

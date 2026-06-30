# ML-vs-LLM-CRCM

This repository is used to compare machine learning (ML) methods with large language models (LLMs) for predicting distant metastasis after colorectal cancer (CRC) surgery. The project focuses on the task of mapping baseline clinical, pathological, immune, and laboratory indicators to a binary prediction of metastasis occurrence (0/1). It includes data preprocessing, statistical analysis, ML modeling, LLM inference experiments, and an interactive prediction application.

## 1. Project Goals

- Compare the performance of traditional machine learning models and large language models for CRC metastasis prediction.
- Evaluate how different prompting strategies (zero-shot, few-shot, and prompt augmentation) affect LLM prediction quality.
- Provide reproducible experiment scripts and visualization/analysis scripts for further research and paper reproduction.

## 2. Repository Structure

- LLM_predict/
  - Used for LLM-based experiments.
  - Contains scripts for zero-shot, few-shot, and prompt-augmented inference strategies.
  - Uses model_matrix.json to define the model and script combinations for experiments.

- ML_predict/
  - Used for the machine learning pipeline.
  - Includes outlier processing, statistical analysis, modeling, interpretability analysis, and a Streamlit-based demo application.

## 3. Main Scripts

### LLM_predict

- run_all_experiments.py
  - Runs all configured LLM experiments in batch.
- 1_zero_shot_naive.py
  - Basic zero-shot inference.
- 2_few_shot.py
  - Few-shot prompting inference.
- 3_prompt_aug.py
  - Inference using augmented prompts.
- 4_mix_few_aug.py
  - Hybrid few-shot and augmented prompting strategy.
- 0_4_types_inputs.py
  - Evaluates the effect of different input formats on LLM results.
- model_matrix.json
  - Configures the models and script list used in the experiments.

### ML_predict

- 1-process_outliers.py
  - Outlier handling.
- 2-analyse_all.py
  - Statistical analysis and group comparisons.
- 3-mice-ternal.py
  - Missing value imputation.
- 4-unit&multi_analyse.py
  - Univariate and multivariate analysis.
- 5-rcs.py
  - Restricted Cubic Spline (RCS) analysis.
- 6-Elastic Net+model+SHAP-ternal+external.PY
  - Elastic Net + model training + SHAP analysis.
- 7-streamlit_app.py
  - Interactive prediction application built with Streamlit.

## 4. Environment Requirements

Python 3.9+ is recommended.

Common dependencies include:

- pandas
- numpy
- scipy
- scikit-learn
- streamlit
- openai
- joblib

You can install the main dependencies with:

```bash
pip install pandas numpy scipy scikit-learn streamlit openai joblib
```

## 5. Quick Start

### 5.1 Run LLM Experiments

Set the API key first (example for DashScope):

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

On Windows PowerShell, use:

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

Then run the batch experiments:

```bash
python LLM_predict/run_all_experiments.py --config LLM_predict/model_matrix.json
```

### 5.2 Launch the ML Prediction Interface

```bash
streamlit run ML_predict/7-streamlit_app.py
```

## 6. Data Notes

The scripts in this repository assume that the relevant input data files are already available locally, such as:

- LLM experiment data (JSON, CSV, Excel, etc.)
- Training and testing data for machine learning modeling
- Intermediate result directories and model files if required

If your data files are stored in different paths from those defined in the scripts, you will need to update the corresponding input paths.

## 7. Output Results

LLM-related scripts typically save results as JSON files for subsequent statistics, aggregation, and plotting. The machine learning scripts generate summary tables, model results, SHAP plots, and other visual outputs.

## 8. Notes

- This project is mainly intended for research reproduction and experimental analysis.
- Some scripts depend on specific data files or model weight files and must be configured according to your local environment.
- To extend the study further, you can adjust the model and script list in model_matrix.json.

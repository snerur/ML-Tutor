# ML Fairness Studio

> An interactive, end-to-end platform for building, evaluating, and auditing machine learning models through the lens of fairness and responsible AI.

---

## About

**ML Fairness Studio** was developed by **Sridhar Nerur** for use in a graduate-level course on responsible machine learning, with substantial assistance from [Claude](https://www.anthropic.com/claude) (Anthropic's AI assistant). The application is built with [Streamlit](https://streamlit.io/) and covers the full ML pipeline — from raw data ingestion to fairness auditing and causal inference — in a single, browser-based interface.

> **Educational Use Only.** This tool is designed to help students and practitioners explore algorithmic fairness interactively. Outputs should always be validated critically and tested for consistency and reproducibility before drawing conclusions.

---

## Features

| Step | Module | What it does |
|------|--------|--------------|
| 1 | 📂 **Data Upload** | Load CSV / Excel files or choose a built-in sample dataset; set the target column and protected attributes |
| 2 | ⚙️ **Preprocessing** | Impute missing values, encode categoricals, scale features, handle class imbalance; automated feature selection via SelectKBest, Lasso (L1), RFE, or PCA |
| 3 | 🔍 **Bias Detection** | Visualize demographic distributions and flag statistical bias in protected attributes *before* training |
| 4 | 🏋️ **Model Training** | Train one of nine ML algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN, Naive Bayes) with cross-validation and hyperparameter tuning |
| 5 | ⚖️ **Fairness Evaluation** | Measure demographic parity difference/ratio, equalized odds, equal opportunity, and disparate impact using [Fairlearn](https://fairlearn.org/) |
| 6 | 🧪 **Model Testing** | Confusion matrix, ROC curves, precision-recall curves, and full classification reports on a held-out test set |
| 7 | 📊 **Feature Importance** | SHAP values, permutation importance with confidence intervals and p-values |
| 8 | 🤖 **LLM Analysis** | Chat with an AI assistant (OpenAI, Anthropic Claude, Google Gemini, Ollama, or any OpenAI-compatible endpoint) for pipeline insights and recommendations |
| 9 | 🔗 **Causal Inference** | Estimate average treatment effects with IPW, AIPW, and propensity-score methods |
| 10 | 📓 **Download Notebook** | Export the entire pipeline as a self-contained, runnable Jupyter Notebook |

---

## Getting Started

### Prerequisites

- Python **3.9 – 3.11** (NumPy 2.x is not yet fully supported by all dependencies; prefer 3.10)
- `pip` or `conda`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/snerur/ML-Tutor.git
cd ML-Tutor

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## Walkthrough

1. **Load data** — Navigate to *Data Upload* in the sidebar. Upload your own CSV/Excel file or pick a sample dataset (e.g., UCI Adult Income, COMPAS). Select the **target column** and any **protected attributes** (e.g., `sex`, `race`, `age_group`).

2. **Preprocess** — Choose imputation strategies, encoding methods (one-hot, ordinal, target), and scaling. The app tracks your choices and replicates them faithfully in the exported notebook.

3. **Detect bias** — Review distributional plots and statistical tests that reveal whether protected groups are over- or under-represented in the data *before* a single model is trained.

4. **Train a model** — Select an algorithm, tune hyperparameters through the UI, and run cross-validated training. The Pipeline Progress tracker in the sidebar keeps you oriented.

5. **Evaluate fairness** — After training, jump to *Fairness Evaluation* to see whether the model treats demographic groups equitably. Multiple fairness criteria are reported side-by-side so you can reason about trade-offs.

6. **Dig deeper** — Use *Feature Importance* to understand which variables drive predictions, *LLM Analysis* to get an AI second opinion, and *Causal Inference* to move beyond correlation.

7. **Export** — Download a fully annotated Jupyter Notebook that reproduces every step, so analyses are auditable and shareable.

---

## LLM Assistant Setup

The AI assistant tab supports several providers. Configure the provider, model, and API key in the **AI Assistant** panel in the sidebar.

| Provider | Notes |
|----------|-------|
| **OpenAI** | Requires an OpenAI API key |
| **Anthropic Claude** | Requires an Anthropic API key |
| **Google Gemini** | Requires a Gemini API key |
| **Ollama (Local)** | Run models locally; no API key needed. Start Ollama first: `ollama serve` |
| **Custom (OpenAI-compatible)** | Point to any OpenAI-compatible endpoint with a custom base URL |

---

## A Note on AI-Assisted Development

This application was built collaboratively with Claude, Anthropic's AI assistant. The development process is itself a case study in human–AI partnership: the domain knowledge, pedagogical goals, and design decisions came from the instructor; Claude contributed code generation, debugging, and documentation support.

This mirrors how AI tools are increasingly used in practice — as accelerators that still require human direction, critical review, and domain expertise. No AI output should be treated as ground truth, and that applies equally to the outputs of the models you train inside this app.

---

## Caveats & Responsible Use

- Results may vary across runs due to random seeds, data splits, and stochastic algorithms. Always test for reproducibility.
- Fairness metrics are mathematical proxies for complex social phenomena. A model that satisfies one fairness criterion may violate another. Interpret metrics in context.
- The LLM assistant can produce confident-sounding but incorrect analyses. Cross-check AI-generated insights against the underlying statistics.
- This tool is not validated for production or high-stakes decision-making. It is a learning instrument.

---

## License

This project is released for educational use. Please cite appropriately if you adapt it for your own courses or research.

---

*Built with ❤️ by Sridhar Nerur · Powered by Streamlit, scikit-learn, Fairlearn, SHAP, and Claude*

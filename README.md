# Fake-News-Detection-NLP-ML-
# Fake News Detection — TF-IDF + Naive Bayes / Logistic Regression


**Overview**
A simple, reproducible pipeline to classify English news/headlines as *fake* or *real* using TF-IDF features and classical ML models. Good baseline for NLP portfolios and Kaggle-style notebooks.


**Files**
- `Fake_News_TF-IDF_Notebook.ipynb` — Jupyter notebook (cells included in the project).
- `train_and_eval.py` — (optional) single-file script to train and save models.
- `streamlit_app.py` — basic demo app for predictions.
- `distilbert_colab.py` — Colab-ready transformer training script.
- `README.md` — this file.


**How to run (local / Kaggle)**
1. Install: `pip install pandas scikit-learn joblib streamlit matplotlib seaborn`
2. Place dataset at `/mnt/data/english_fake_news_2212.csv` or edit the `DATA_PATH` variable.
3. Open the notebook and run cells in order. Or run `python train_and_eval.py`.
4. After training, models are saved to `/mnt/data/fake_news_output/`.
5. Run the demo: `streamlit run streamlit_app.py --server.port 8501`.


**Evaluation**
- Look at `metrics_report.txt` for classification reports.
- Use F1 as the main metric (handles class imbalance better).


**Improvements & next steps**
- Transformer-based model (script included). Use GPU in Colab for training.
- Class balancing: upsample/SMOTE or class weights.
- Model explainability: LIME/SHAP for top tokens.
- Ensembling TF-IDF models + transformer embeddings.


**Author**
Your name — add dataset description and references here.

# Sentiment Classification: ML and DL Approaches

This project demonstrates sentiment classification on movie reviews using two parallel approaches:
- **Traditional Machine Learning:** TF-IDF features + Logistic Regression (with hyperparameter search)
- **Deep Learning:** Keras Bidirectional LSTM on tokenized/padded text (with hyperparameter tuning and early stopping)

All code is provided as clean, reproducible Jupyter notebooks, ready for use in VS Code or Jupyter Lab.

---

## Setup Instructions

1. **Clone the Repository**
    ```bash
    git clone https://github.com/AnthropoidFHJ/Sentiment-Classification-ML-DL
    cd Sentiment-Classification-ML-DL
    ```

2. **Create Python Environment**
    ```bash
    python -m venv venv

    # Linux/macOS:
    source venv/bin/activate
    
    # Windows:
    venv\Scripts\activate
    ```

3. **Install Dependencies**
    - You can run the first cell in each notebook to install all required packages, or install manually:
    ```bash
    pip install pandas numpy scikit-learn nltk joblib matplotlib tensorflow
    ```

4. **Prepare Data**
    - Place your data files in the `Data/` folder:
        - `Data/train_data.csv`
        - `Data/test_data.csv`
    - The first two columns should be text and label. If named differently, the notebooks will attempt to rename automatically.

5. **Run the Notebooks**
    - Open `Approaches/ML_Approach.ipynb` & `Approaches/DL_Approach.ipynb` in VS Code or Jupyter.
    - Run all cells from top to bottom.

---

## Approach

- **ML Pipeline:**  
  Clean text (lowercase, remove HTML, numbers, punctuation, stopwords) → TF-IDF → Logistic Regression → GridSearchCV for hyperparameters → Evaluation.
- **DL Pipeline:**  
  Clean text → Tokenize → Pad sequences → Bidirectional LSTM (Keras) → Hyperparameter tuning (embedding size, LSTM units, batch size, early stopping) → Evaluation.

---

## Tools Used

- **Python** (Jupyter/VS Code)
- **pandas, numpy** (data handling)
- **scikit-learn** (TF-IDF, Logistic Regression, metrics, GridSearchCV)
- **NLTK** (stopwords)
- **TensorFlow/Keras** (tokenization, padding, LSTM)
- **matplotlib** (plots, confusion matrix)
- **joblib** (save/load models and vectorizers)

---

## Results

- **ML (TF-IDF + Logistic Regression):**
    - Typical accuracy: **~88%** on IMDB-style test set (25,000 samples)
    - Balanced precision/recall/F1
    - Confusion matrix and metrics printed in notebook
    - Artifacts saved: `Files/model.pkl`, `Files/vectorizer.pkl`

- **DL (Bidirectional LSTM):**
    - Typical accuracy: **~83-86%** (with tuning and regularization)
    - Early stopping used in hyperparameter search to prevent overfitting
    - Training/validation loss and accuracy curves plotted
    - Artifacts saved: `Files/lstm_model.h5`, `Files/tokenizer.joblib`, `Files/encoder.joblib`

---

## Artifacts

- All trained models and vectorizers/tokenizers are saved in the `Files/` directory for easy reuse.

---

## Notes

- If NLTK stopwords are not available, the notebooks download them at runtime.
- If your CSVs are already split into train/test, the notebooks will use them as provided.
- For larger datasets, consider increasing epochs and tuning hyperparameters further for the LSTM model.
- Interactive prediction cells are included in both notebooks for quick testing.

---

## Author

[AnthropoidFHJ](https://github.com/AnthropoidFHJ)  
Name : Ferdous Hasan  
Date: August 16, 2025
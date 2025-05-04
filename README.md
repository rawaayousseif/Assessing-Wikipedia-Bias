# Assessing Wikipedia Bias

This project aims to detect and score linguistic bias in Wikipedia articles using Natural Language Processing (NLP) and supervised machine learning. The goal is to promote more objective information presentation on publicly editable platforms.

---

## Project Objectives

1. Build a classification system to detect sentence-level bias in Wikipedia articles.
2. Score articles on an article-level bias scale using sentence predictions.
3. Evaluate and compare traditional ML and Transformer-based models (e.g., CatBoost, BERT).
4. Provide a public tool that helps visualize bias in real time.
5. Analyze bias patterns across categories such as Politics, Science, Religion, and Economy.

---

## Methodology

- **Data Preprocessing:** Cleaning text, removing stopwords, emojis, and non-linguistic symbols.
- **Exploratory Data Analysis (EDA):** Word frequency, sentiment analysis (VADER), sentence length, article length.
- **Model Training:** Logistic Regression, XGBoost, CatBoost, and BERT were trained using TF-IDF and transformer encodings.
- **Article Scoring:** Averaged sentence bias probabilities to score full articles.
- **Visualization:** Created dashboards, heatmaps, and model comparisons to showcase bias patterns.

---

## Key Results

- **Best Model:** CatBoost (Accuracy: 77%, AUC: 0.85)
- **Bias Distribution:**
  - Biased articles: 62%
  - Unbiased articles: 38%
- **Top 3 Most Biased Categories:**
  - Science: 0.686
  - Economy: 0.683
  - Social Issues: 0.676
- **Top 3 Least Biased Topics:**
  - Impeachment: 0.58
  - Democratic Party: 0.65
  - Bible: 0.66

---

## Recommendations for Wikipedia

Based on our findings, we recommend the following to Wikipedia editorial and moderation teams:

1. **Implement NLP-Powered Review Tools:** Integrate automated bias detection for editorial review.
2. **Use Transformer Models for Real-Time Monitoring:** BERT or RoBERTa can help flag biased phrases as content is written or edited.
3. **Encourage More Balanced Edits:** Highlight articles with skewed sentiment or word usage for community review.
4. **Increase Multilingual Audits:** Extend bias detection to non-English Wikipedia pages using multilingual models (e.g., XLM-R).
5. **Expand Training Data with Community Labels:** Introduce crowd-sourced flagging of biased articles to improve model accuracy and fairness.

---

## Deployment Plan

We plan to deploy a user-friendly app on GitHub using:

- **Streamlit Interface** for interactive bias scoring
- **Hugging Face Transformers** for model backend
- **Hosted on GitHub**: [Project Repository](https://github.com/rawaayousseif/Assessing-Wikipedia-Bias)

Users can paste a Wikipedia article and instantly view its bias score and sentence-level insights.

---

## Technologies Used

- Python (Pandas, Scikit-learn, NLTK)
- CatBoost & XGBoost
- Hugging Face Transformers (BERT)
- Streamlit
- Matplotlib & Seaborn

---

## File Structure

Assessing-Wikipedia-Bias/
│
├── data/ # Raw and cleaned Wikipedia datasets
├── notebooks/ # EDA and model development
├── models/ # Saved CatBoost and BERT models
├── app/ # Streamlit web app
├── utils/ # Preprocessing, tokenization
├── README.md # Project documentation

## Contributors

- Rawaa Yousseif (Data Scientist)
- Viktor Kliufinskyi (Data Scientist)
- Hamed TB (Data Scientist)

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/rawaayousseif/Assessing-Wikipedia-Bias.git
   cd Assessing-Wikipedia-Bias

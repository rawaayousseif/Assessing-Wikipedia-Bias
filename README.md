# Assessing-Wikipedia-Bias
Bias Detection in Wikipedia Articles

Introduction

Wikipedia is one of the world’s most visited sources of publicly editable information. While its open-access model promotes inclusivity and rapid knowledge dissemination, it also introduces risks of linguistic bias. This project aimed to detect such bias using Natural Language Processing (NLP) and supervised machine learning, offering an article-level and sentence-level assessment of potential bias.

Objectives

Identify linguistic bias in Wikipedia articles.

Build machine learning models that classify sentences or articles as biased or unbiased.

Score entire articles using averaged sentence-level predictions.

Evaluate real-world Wikipedia articles across controversial and neutral topics.

Methodology

Data Collection: We used a labeled dataset (articles.csv) containing Wikipedia articles tagged as biased or unbiased.

EDA: We examined class distributions, text length, sentiment scores, and token patterns.

Preprocessing: Texts were cleaned, stopwords removed, and vectorized using TF-IDF.

Modeling: Multiple classifiers (Logistic Regression, XGBoost, CatBoost, BERT) were trained and evaluated.

Bias Scoring: Sentence-level predictions were averaged to assign an overall article bias score.

Evaluation: Real Wikipedia articles were scored and analyzed across categories (Politics, Science, Religion, etc.).

Deployment: The model is being prepared for deployment on GitHub and a potential Streamlit app.

Exploratory Data Analysis (EDA)

Dataset Overview

Total articles: 2,095

Labeled entries: 2,094

Biased: 1,294 (62%)

Unbiased: 800 (38%)

Article Length (words)

Mean: 484

Median: 304

Min: 1

Max: 5,828

Distribution: Skewed with many short entries

Sentence Count per Article

Most articles had 1–3 sentences using NLTK's punkt tokenizer.

Sentiment Analysis (VADER)

Biased articles had more extreme sentiment polarity.

Unbiased articles were closer to neutral sentiment.

Top Words by Class

Biased: clinton, trump, said, hillary, campaign

Unbiased: media, get, election, campaign

Model Performance

Model

Accuracy

Recall (Unbiased)

Recall (Biased)

ROC-AUC

Logistic Regression

72%

60%

79%

0.81

XGBoost

78%

56%

91%

0.83

CatBoost

77%

61%

88%

0.85

BERT (avg)

~76%

Balanced

Balanced

~0.85

Bias Scoring Function

def predict_article_bias(article_text):
    sentences = sent_tokenize(article_text)
    cleaned = [clean_text(s) for s in sentences]
    vec = vectorizer.transform(cleaned)
    probs = cat_model.predict_proba(vec)[:, 1]
    return round(probs.mean(), 2)

Scores closer to 1.0 suggest stronger bias; closer to 0.0 suggests neutrality.

Large-Scale Topic Evaluation

Categories:

Politics: Biden, Trump, NATO, Impeachment

Science: Evolution, CRISPR, Vaccines, Big Bang

Religion: Islam, Bible, Atheism

Economy: Inflation, Capitalism, Cryptocurrency

Social Issues: Abortion, Racism, Gender Identity

Average Bias Score by Category:

Category

Bias Score

Science

0.686

Economy

0.683

Social Issues

0.676

Religion

0.669

Politics

0.650

Least Biased Articles:

Topic

Score

Category

Impeachment

0.58

Politics

Bible

0.66

Religion

Inflation

0.67

Economy

Conclusion

This project successfully built a machine learning pipeline to detect bias in Wikipedia articles. With over 2,000 labeled articles analyzed, the best-performing model (CatBoost) achieved 0.85 AUC. Further validation was performed using real articles on controversial and neutral topics. The pipeline includes both sentence-level and article-level prediction mechanisms.

We also integrated a BERT-based model and found that it yields consistent results with improved contextual understanding. Bias scoring proved useful for comparing content across themes like science, politics, and religion.

Strategic Recommendations for Wikipedia

As data scientists, we propose the following improvements to Wikipedia’s editorial pipeline based on our analysis:

Model Integration: Deploy a real-time bias detector using CatBoost or BERT to flag linguistically biased edits before publication.

Editorial Dashboard: Build a tool to help editors visualize sentence-level bias and highlight polarizing language.

Reviewer Alerts: Flag newly edited articles with high predicted bias for human moderation.

Language Expansion: Extend bias detection models to cover multilingual Wikipedia editions.

Dataset Growth: Create a crowdsourced labeling system to expand the biased/unbiased corpus beyond 10K entries.

Neutrality Training: Use flagged sentences to build a training set for neutral rewording suggestions via LLMs.

Transparency via Explainability: Use SHAP or LIME to explain model predictions and foster trust among contributors.

Repository and Contributors

The full codebase, notebooks, and scoring scripts are available on GitHub:

🔗 https://github.com/rawaayousseif/Assessing-Wikipedia-Bias

Team Members

Rawaa Yousseif

Viktor Kliufinskyi

Hamed TB


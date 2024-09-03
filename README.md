# 🌟 Machine Learning NLP: Drugs Reviews

Our project aims to leverage advanced techniques in natural language processing (NLP) and machine learning (ML) to analyze and extract valuable insights from a dataset containing medication reviews. The objective is to understand the relationship between patients' opinions on medications and the quantitative assessments of their effectiveness and side effects. Additionally, we will incorporate various other variables to enhance the reliability and realism of our results, thereby gaining a comprehensive understanding of the impact of different drugs on various patients.

---

## 🗂️ Project Overview

---

### 1. 📚 Text Preprocessing and Vectorization

**Text Preprocessing**:
- **Objective**: Prepare text data for analysis by reducing noise and variability.
- **Steps**: Transform raw text into a structured format to facilitate feature extraction. This involves cleaning and normalizing text, such as removing irrelevant characters and standardizing terms.
- **Outcome**: Structured text data ready for vectorization.

**Vectorization**:
- **Objective**: Convert preprocessed text into numerical representations usable by machine learning models.
- **Methods**:
  - **Bag-of-Words (BoW) and TF-IDF**: Represent text by counting word frequencies and term importance.
  - **Word2Vec/Glove**: Generate word embeddings based on context.
  - **Doc2Vec**: Create document-level embeddings by considering the entire document context.
  - **LDA (Latent Dirichlet Allocation)**: Extract topics from the text corpus and represent documents as topic distributions.
- **Preparation**: Clean dataset by removing infrequent and overly common words to refine the vector representations.

---

### 2. 🤖 Machine Learning Models

**Classification**:
- **Objective**: Predict the effectiveness of drugs based on review types (benefits, side effects, comments).
- **Approach**:
  - **Models Tested**: K-Nearest Neighbors (KNN) and Support Vector Classification (SVC).
  - **Process**: Apply cross-validation to select optimal models and configurations for each type of review.
  - **Comparison**: Evaluate model performance using different vectorization techniques (BoW, TF-IDF, Word2Vec, LDA).

**Regression**:
- **Objective**: Predict drug ratings based on review text using various regression models.
- **Models Tested**: Linear Regression, Random Forest, Gradient Boosting, and Support Vector Regression (SVR).
- **Techniques**: Utilize vectorization methods (BoW, TF-IDF, Word2Vec) and dimensionality reduction (e.g., Singular Value Decomposition) to improve prediction accuracy.
- **Process**: Compare model performance with and without dimensionality reduction to assess impact on prediction.

---

### 3. 📊 Dashboard

**Objective**: Provide an interactive tool for visualizing and analyzing the dataset and model performance.
- **Tabs**:
  - **Tab 1: LDA Topic Visualization**: Interactive charts and pyLDAvis visualizations showing topic distribution in the corpus.
  - **Tab 2: Topic-Document Similarity Heatmap**: Matrix visualizing document-topic probabilities, helping to understand document distribution across topics.
  - **Tab 3: Classification Model Evaluation**: Performance graphs for SVC and KNN, allowing users to explore model accuracy across different configurations and datasets.
  - **Tab 4: Regression Hyperparameter Tuning**: Visualization of RMSE impacts from hyperparameter adjustments in various regression models.

---

### 📄 Conclusions
Detailed conclusions and insights from this project are available in the .pdf written report


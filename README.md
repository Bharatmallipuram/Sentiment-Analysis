# Sentiment Analysis Project  

## 📌 Overview  
This project focuses on developing a **sentiment analysis model** using **Natural Language Processing (NLP)** techniques. The goal is to classify **movie reviews** as **Positive** or **Negative** based on their text content. The model is built using **Python, TensorFlow, and Keras**, with **TF-IDF** for feature extraction.

---

## 📂 Table of Contents  
- [Dataset](#dataset)  
- [Tools & Technologies](#tools--technologies)  
- [Setup Instructions](#setup-instructions)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Training](#model-training)  
- [Sentiment Prediction](#sentiment-prediction)  
- [Evaluation & Results](#evaluation--results)  
- [Future Work](#future-work)  
- [Conclusion](#conclusion)  

---

## 📊 Dataset  
- **Source**: IMDB dataset containing **50,000** movie reviews.  
- **Columns**:  
  - `review`: Text of the movie review.  
  - `sentiment`: Sentiment label (**Positive** or **Negative**).  

---

## 🛠️ Tools & Technologies  
- **Programming Language**: Python  
- **Libraries**:  
  - `pandas`, `NumPy`, `re`, `NLTK`, `Scikit-learn` (for data preprocessing)  
  - `TensorFlow`, `Keras` (for deep learning model)  
  - `Joblib` (for model saving)  
  - `Streamlit` *(optional: for UI deployment)*  
- **Environment**: Jupyter Notebook for development, Streamlit for deployment *(optional)*  

---

## ⚙️ Setup Instructions  

### 🔹 Prerequisites  
Before running the project, ensure you have the following installed:  
- **Python 3.7+**  
- **Jupyter Notebook**  
- **Required Python Libraries** (install using the command below):  
  ```bash
  pip install pandas numpy nltk scikit-learn tensorflow keras joblib streamlit

### 🔹 Directory Structure
```bash
Sentiment-Analysis/
│── data/                 # Dataset files  
│── models/               # Saved trained model  
│── notebooks/            # Jupyter notebooks  
│── scripts/              # Python scripts for training and prediction  
│── app.py                # Streamlit web app (optional)  
│── README.md             # Project documentation  
```
### 🔄 Data Preprocessing

1. Data Cleaning:
Remove HTML tags
Remove punctuation
Convert text to lowercase

2. Tokenization:
Split sentences into individual words using NLTK’s word_tokenize()

3. Stopwords Removal:
Remove common stopwords like "and", "the", "is", etc.

4. Stemming:
Convert words to their root form (e.g., "running" → "run")

5. TF-IDF Vectorization:
Convert text data into numerical vectors using Scikit-learn’s TfidfVectorizer

## 🏗️ Model Training
### 🔹 Model Architecture

The model is a feedforward neural network with the following layers:

Input Layer: Dense (128 units, ReLU activation)
Hidden Layers: Two Dense layers (64 & 32 units, ReLU activation)
Output Layer: Dense (2 units, Sigmoid activation for binary classification)

### 🔹 Compilation & Training
Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy
Training:
80% of data for training
20% of data for testing
Trained for 10 epochs

### 🔹 Saving the Model
``` bash
import joblib
joblib.dump(model, "models/sentiment_model.pkl")
```

### 📌 Sentiment Prediction
Script Overview (predict.py)
Loads the trained model and TF-IDF vectorizer
Processes new input text
Outputs the predicted sentiment

### Usage
Run the script and input a review:

```bash
python scripts/predict.py
```
Example:
```bash
Input: "This movie was fantastic! The acting was brilliant."
Output: "Predicted Sentiment: Positive"
```

### 🌐 Streamlit Web UI (Optional)
The project can be deployed as a web application using Streamlit.

#### Run the Web App
```bash
streamlit run app.py
``` 

### 📊 Evaluation & Results
🔹 Model Performance
The model achieves an accuracy of 74% on the test dataset.
Metrics like Precision, Recall, and F1-score are used for evaluation.
🔹 Confusion Matrix
A confusion matrix is plotted to visualize the model’s classification performance.

### 🚀 Future Work
Hyperparameter Tuning: Improve accuracy by experimenting with different architectures.
Data Augmentation: Handle class imbalance using augmentation techniques.
Model Deployment: Deploy the model to a cloud platform for real-time sentiment analysis.

### 🏁 Conclusion
This project successfully demonstrates how to build and deploy a sentiment analysis model using machine learning and NLP techniques. The pipeline, from data preprocessing to model training and prediction, is efficiently handled using Python and its libraries.

# 📄 Resume Screening for Python Developer (TF-IDF + Logistic Regression)

## 📌 Overview

This project screens resumes to identify candidates suitable for the role of **Python Developer** using:

- Text preprocessing and cleaning  
- TF-IDF vectorization (unigrams + bigrams)  
- Logistic Regression classifier  

It also visualizes:

- Model performance via confusion matrix  
- Most indicative keywords for suitability  

Users can input resume text to get real-time predictions.

---

## 🚀 Key Features

✔ Text cleaning (lowercase, remove special characters)  
✔ TF-IDF feature extraction  
✔ Binary classification (Suitable / Not Suitable)  
✔ Logistic Regression model  
✔ Accuracy evaluation  
✔ Confusion matrix visualization  
✔ Top keywords analysis  
✔ Interactive resume prediction  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Regex  

---

## 📂 Dataset

The script expects a CSV file:

```
gpt_dataset.csv
```

Required columns:

- `Resume` → Text of the candidate’s resume  
- `Category` → Job category applied for  

The model classifies resumes as **Suitable** for `"Python Developer"` or **Not Suitable** for other categories.

---

## 🔎 Project Workflow

### 1️⃣ Labeling

Resumes are labeled as:

- `1` → Suitable (target job: Python Developer)  
- `0` → Not Suitable (other jobs)  

---

### 2️⃣ Text Cleaning

- Convert to lowercase  
- Remove numbers and special characters  
- Keep only alphabetic characters  

```python
clean_text()
```

---

### 3️⃣ Train-Test Split

- 80% training  
- 20% testing  
- Stratified to maintain class balance  

---

### 4️⃣ TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
```

- Uses unigrams + bigrams  
- Removes English stopwords  
- Limits vocabulary to 5000 features  

---

### 5️⃣ Model Training

Model used:

```
Logistic Regression
```

- Maximum iterations: 1000  
- Binary classification problem  

---

### 6️⃣ Evaluation

Metrics:

- Accuracy  
- Confusion matrix (visualized using seaborn heatmap)  
- Top 10 words contributing to “Suitable” and “Not Suitable” predictions  

---

## 🔮 Interactive Prediction

Type any resume text to predict suitability:

```
Enter resume text (or type 'exit' to quit):
```

Returns:

- "Suitable" → Candidate matches Python Developer role  
- "Not Suitable" → Candidate does not match  

---

## 📦 Installation

Install required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Make sure `gpt_dataset.csv` is in the same directory.

---

## 🎯 Use Cases

- Resume screening automation  
- Candidate shortlisting for Python Developer roles  
- Keyword analysis for hiring  
- NLP text classification practice  
- HR analytics  

---

## 📈 What This Project Demonstrates

- Text preprocessing for resumes  
- TF-IDF feature engineering  
- Logistic Regression for binary classification  
- Model interpretability with top keywords  
- Interactive system for candidate evaluation  

---

## 👨‍💻 Author

Built as part of NLP and AI-based recruitment automation experimentation.

If you found this useful, consider starring the repository ⭐

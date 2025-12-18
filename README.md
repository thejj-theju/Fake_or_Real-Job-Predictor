#  ML-Based Fake Job Postings Detector Using NLP and Machine Learning

A simple and effective text classification project to detect fake job postings using natural language processing and machine learning models, deployed as a Flask web app.

---

## 🧠 Project Flow

The project follows this pipeline:

1. **Importing Libraries**  
2. **Data Loading & Understanding the Data**  
3. **EDA (Handling Missing Values)**  
4. **Text Preprocessing**  
   - Convert to lowercase  
   - Remove punctuation & stopwords  
   - Tokenize text  
5. **Feature Extraction**  
   - Use **TF-IDF** to convert text into numeric features  
6. **Train/Test Split**  
7. **Model Training**  
8. **Model Evaluation**  
9. **Test with New Samples**
10. **Shap(SHapley Additive exPlanations)**
    - helps to  understand why a machine learning model made a particular prediction.
    - SHAP helps make our ML model explainable by showing which features push predictions higher or lower


---

##  Results Summary

| Model                | Accuracy | Precision | Recall   | F1       |
|----------------------|----------|-----------|----------|----------|
| Logistic Regression  | 0.977629 | 0.733668  | 0.843931 | 0.784946 |
| SVM                  | 0.981823 | 0.957627  | 0.653179 | 0.776632 |
| XGBoost              | 0.981823 | 0.813953  | 0.809249 | 0.811594 |
| Naive Bayes          | 0.963367 | 0.956522  | 0.254335 | 0.401826 |
| Decision Tree        | 0.958613 | 0.564767  | 0.630058 | 0.595628 |

---


## 🛠 Technologies Used

- Python  
- scikit-learn  
- Flask  
- pandas, numpy  
- TF-IDF vectorizer  
- (Optional: XGBoost if used)
- Shap (Explainable AI)
---

## 📌 To Run

Use Terminal:

1. Install everything — run the command below:

```bash
pip install -r requirements.txt


2.Execute the file — run the command below:

python app.py
```
---


## Sample Screenshot:
1. How the UI looks.

<img width="1843" height="831" alt="Screenshot 2025-12-18 095557" src="https://github.com/user-attachments/assets/5dd76d3b-e0b4-4513-b230-24e821ddc94e" />

 
2.Enter the text to predict:
<img width="1751" height="803" alt="Screenshot 2025-12-18 095647" src="https://github.com/user-attachments/assets/04b5d289-3cdf-4f33-9076-5e913a0046e7" />

3. Here we can see the words pushes to predict the result:
<img width="1624" height="816" alt="Screenshot 2025-12-18 095709" src="https://github.com/user-attachments/assets/aca7a12f-0877-4a32-8839-5ee2933294d5" />

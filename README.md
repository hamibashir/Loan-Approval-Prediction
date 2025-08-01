# 🏦 **Loan Approval Prediction System**

A machine learning-based classification project designed to predict loan approval outcomes using applicant information. Built using **Logistic Regression** and **Decision Tree** algorithms, with imbalanced data handled via **SMOTE**. The project includes data preprocessing, modeling, and visual performance evaluation.

---

📌 **Project Overview**

Predicting whether a loan application will be approved is a common task in banking and finance. This project utilizes historical loan application data to train models that can classify future applications as **approved** or **rejected** based on applicant features such as income, employment status, and education.

By leveraging machine learning techniques, the project aims to:

* Clean and preprocess real-world loan data
* Handle class imbalance using **Synthetic Minority Over-sampling Technique (SMOTE)**
* Train and evaluate **Logistic Regression** and **Decision Tree** classifiers
* Compare model performance using classification metrics and confusion matrices

---

🧠 **Technologies & Libraries Used**

* **Python**
* **Pandas**
* **NumPy**
* **Seaborn**
* **Matplotlib**
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**

---

⚙️ **Key Features**

* Import and explore raw loan data
* Clean and encode categorical features
* Split dataset into training and testing sets
* Apply **SMOTE** to balance imbalanced classes
* Train two models:

  * **Logistic Regression** for linear decision boundaries
  * **Decision Tree** for hierarchical decision-making
* Generate and compare detailed **classification reports**
* Visualize model performance using **confusion matrix heatmaps**

---

📊 **Model Evaluation & Insights**

* **Classification Report:** Includes Precision, Recall, F1-Score, and Accuracy
* **Confusion Matrix:** Visual comparison of predicted vs actual results
* Performance of both models is analyzed on the same testing data to identify strengths and weaknesses

---

🖼️ **Visualization Highlights**

* Confusion matrix heatmaps for both models
* Color-coded results for quick interpretation of prediction accuracy

---

💻 **Usage Instructions**

1. Install required packages:

   ```
   pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
   ```

2. Replace the dataset path in the script with your local path.

3. Run the script to see the data preprocessing, model training, evaluation, and plots.

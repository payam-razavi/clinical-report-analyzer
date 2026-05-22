\# Phase 1 — Model Development



Developed by Payam Razavi



Clinical NLP pathology report analysis system developed through the  

Cedars-Sinai / Los Angeles Pierce College AI Initiative.



\---



\## Overview



This phase focuses on training machine learning models for  

pathology report classification using clinical text data.



The pipeline converts unstructured pathology reports into  

numerical TF-IDF feature vectors and evaluates multiple  

machine learning algorithms for cancer type prediction.



\---



\## Objectives



\- Preprocess pathology report text

\- Generate TF-IDF feature vectors

\- Train multiple machine learning models

\- Evaluate classification performance

\- Select the best-performing model

\- Save deployment-ready artifacts



\---



\## Models Evaluated



The following machine learning models were tested:



\- Logistic Regression

\- Decision Tree

\- Random Forest



Models were compared using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Confusion Matrix



\---



\## Dataset



TCGA-based pathology report dataset provided through the  

Cedars-Sinai National AI Campus program.



Approximate dataset size:



\- \~9,500 pathology reports



\---


## Dataset Note

The original training dataset is not included in this repository due to size and research-data considerations.

---


\## Output Files



This phase generates:



\- `model.pkl`

\- `vectorizer.pkl`



These files are later used during Phase 2 deployment.



\---



\## Main Notebook



```text

model\_training.ipynb



```



\---



\## Pipeline Overview



!\[Phase 1 Pipeline](phase1\_pipeline.png)



\---



\## Technologies Used



\- Python

\- scikit-learn

\- pandas

\- NumPy

\- TF-IDF Vectorization

\- matplotlib

\- seaborn



\---



## Notes

The final selected model and TF-IDF vectorizer are reused during  
Phase 2 clinical report analysis and deployment.

---

## Model Files Note

Serialized model files (.pkl) are not included in this repository due to file size limitations.

The repository contains the complete training pipeline, preprocessing logic, feature engineering workflow, and deployment code required to reproduce the project.
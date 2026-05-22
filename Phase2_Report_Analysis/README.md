\# Phase 2 — Clinical Report Analysis



Developed by Payam Razavi



Clinical NLP pathology report analysis system developed through the  

Cedars-Sinai / Los Angeles Pierce College AI Initiative.



\---



\## Overview



This phase deploys the trained pathology classification model  

through an interactive Streamlit web application.



The system accepts pathology report PDFs, extracts clinical text,  

and generates explainable machine learning predictions.



\---



\## Features



\- PDF pathology report upload

\- Text extraction and preprocessing

\- Pathology-content validation

\- Cancer type prediction

\- Confidence scoring

\- Keyword extraction

\- Highlighted diagnostic evidence

\- Plain-English report summaries



\---



\## Input



\- Unstructured pathology reports

\- PDF format



The application supports pathology reports from multiple hospitals  

with different formatting styles.



\---



\## Outputs



The application generates:



\- Top predicted cancer types

\- Confidence scores

\- Important keywords

\- Highlighted diagnostic sentences

\- Full report visualization

\- Plain-English summaries



\---



\## Explainability System



The system highlights clinically relevant sections of the report  

to improve interpretability and transparency.



The explainability layer includes:



\- keyword importance scoring

\- diagnostic sentence highlighting

\- prediction confidence analysis

\- plain-English summaries



\---



\## Validation Logic



Additional preprocessing and validation logic were implemented to:



\- reject irrelevant non-medical documents

\- detect pathology templates/sample reports

\- reduce noisy PDF extraction artifacts

\- improve handling of image-heavy reports

\- remove duplicated metadata and disclaimers



\---



\## Files



```text

app.py

model.pkl

vectorizer.pkl

sample\_reports/



```



\---



\## Run the Application



```bash

streamlit run app.py

```



\---



\## Required Packages



```bash

pip install streamlit PyPDF2 joblib scikit-learn

```



\---



\## Pipeline Overview



!\[Phase 2 Pipeline](phase2\_pipeline.png)



\---



\## Sample Reports



The `sample\_reports/` folder contains example PDFs demonstrating:



\- irrelevant document rejection

\- pathology template detection

\- pathology report prediction

\- image-heavy pathology report handling



\---



## Important Note

Phase 2 depends on the trained model and vectorizer  
generated during Phase 1.

The model files must remain in the same folder as `app.py`  
unless file paths are modified.

---

## Model Files Note

Serialized model files (.pkl) are not included in this repository due to file size limitations.

The repository contains the complete training pipeline, preprocessing logic, feature engineering workflow, and deployment code required to reproduce the project.

---

## Disclaimer


This project is intended for educational and research purposes only.  

It is not intended for clinical diagnosis or medical decision-making.


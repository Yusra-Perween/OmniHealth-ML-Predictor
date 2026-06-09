# OmniHealth ML Predictor

OmniHealth ML Predictor is a sophisticated machine learning-based application designed to analyze user health data and provide accurate early-warning predictions for common medical conditions such as Diabetes and Heart Disease. The goal of this project is to demonstrate the practical, end-to-end application of Machine Learning models in modern healthcare systems.

## Project Objectives
- To apply advanced machine learning techniques on complex health-related datasets
- To predict possible medical conditions based on dynamic user input features
- To provide a simple and understandable health recommendation workflow with visual insights

## Key Features
- ML-based disease prediction
- Multiple trained models for different health conditions
- Streamlit-powered interactive web dashboard
- Clean and modular Python code structure

## Technologies Used
- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Joblib
- Plotly & Seaborn (for data visualization)

## Project Structure
- `app.py` – Main Streamlit application
- `train_model.py` & `train_heart_model.py` – Model training logic
- `*.csv` – Datasets used for training and dashboard visualization
- `*.pkl` – Saved trained ML models
- `requirements.txt` – Project dependencies

## How to Run the Project
1. Install dependencies  
   `pip install -r requirements.txt`

2. Run the application  
   `streamlit run app.py`

## Future Enhancements
- Integration with cloud-based database architectures (e.g., AWS, Firebase) for persistent patient history storage.
- Real-time deployment via Docker and cloud hosting platforms.
- Addition of predictive models for other conditions like chronic kidney disease.

## Note
This project is created for learning and academic purposes and is not intended
for real-world medical diagnosis.

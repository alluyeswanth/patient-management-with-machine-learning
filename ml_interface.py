from fastapi import FastAPI, Path, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, List, Dict
import json
from datetime import datetime, timedelta
import os
from faker import Faker
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import the risk assessment system (assuming it's in the same directory)
# If you saved the ML model as a separate file, import it here
# from patient_risk_ml import PatientRiskAssessment

app = FastAPI(title="Enhanced Patient Management System with ML Risk Assessment")
fake = Faker()

# Initialize ML system
ml_system = None  # Will be initialized after first data generation

# Extended disease list for more realistic data
DISEASES = [
    "Hypertension", "Diabetes Type 2", "Asthma", "Arthritis", "Migraine",
    "Depression", "Anxiety", "Back Pain", "Allergies", "Insomnia",
    "High Cholesterol", "Gastritis", "Anemia", "Thyroid Disorder", "Osteoporosis",
    "Heart Disease", "Kidney Stones", "Liver Disease", "Bronchitis", "Sinusitis",
    "Eczema", "Psoriasis", "Ulcer", "Gallstones", "Pneumonia",
    "Tuberculosis", "Malaria", "Dengue", "Typhoid", "Hepatitis",
    "COVID-19", "Influenza", "Common Cold", "Skin Infection", "UTI",
    "Constipation", "Diarrhea", "Food Poisoning", "Dehydration", "Fatigue",
    "Vitamin D Deficiency", "Iron Deficiency", "Calcium Deficiency", "None", "Healthy"
]

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

class Patient(BaseModel):
    id: Annotated[str, Field(..., description='ID of the patient', examples=['P001'])]
    name: Annotated[str, Field(..., description='Name of the patient')]
    city: Annotated[str, Field(..., description='City where the patient is living')]
    age: Annotated[int, Field(..., gt=0, lt=120, description='Age of the patient')]
    gender: Annotated[Literal['male', 'female', 'others'], Field(..., description='Gender of the patient')]
    height: Annotated[float, Field(..., gt=0, description='Height of the patient in mtrs')]
    weight: Annotated[float, Field(..., gt=0, description='Weight of the patient in kgs')]
    phone: Annotated[str, Field(..., description='Phone number of the patient')]
    email: Annotated[str, Field(..., description='Email address of the patient')]
    blood_group: Annotated[str, Field(..., description='Blood group of the patient')]
    disease: Annotated[str, Field(..., description='Primary disease/condition of the patient')]
    visits: Annotated[List[str], Field(default_factory=list, description='List of visit timestamps')]

    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 2)
        
    @property
    def verdict(self) -> str:
        if self.bmi < 18.5:
            return 'Underweight'
        elif self.bmi < 25:
            return 'Normal'
        elif self.bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

    def as_dict(self):
        d = self.dict()
        d['bmi'] = self.bmi
        d['verdict'] = self.verdict
        return d

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    age: Optional[int] = Field(default=None, gt=0)
    gender: Optional[Literal['male', 'female', 'others']] = None
    height: Optional[float] = Field(default=None, gt=0)
    weight: Optional[float] = Field(default=None, gt=0)
    phone: Optional[str] = None
    email: Optional[str] = None
    blood_group: Optional[str] = None
    disease: Optional[str] = None

class BulkGenerateRequest(BaseModel):
    count: Annotated[int, Field(..., gt=0, le=100000, description='Number of patients to generate (max 100,000)')]
    clear_existing: Annotated[bool, Field(default=False, description='Whether to clear existing data')]

class MLTrainingRequest(BaseModel):
    retrain: Annotated[bool, Field(default=False, description='Whether to retrain the model')]

# Embedded ML Risk Assessment System
class PatientRiskAssessment:
    def __init__(self):
        self.risk_model = None
        self.treatment_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def create_risk_labels(self, df):
        """Create risk and treatment labels based on patient data"""
        risk_labels = []
        treatment_labels = []
        
        for _, patient in df.iterrows():
            age = patient.get('age', 0)
            bmi = patient.get('bmi', 25)
            disease = str(patient.get('disease', 'None')).lower()
            visit_freq = patient.get('visit_frequency', 0)
            
            risk_score = 0
            
            # Age factor
            if age > 65:
                risk_score += 3
            elif age > 50:
                risk_score += 2
            elif age < 18:
                risk_score += 1
            
            # BMI factor
            if bmi < 18.5 or bmi > 30:
                risk_score += 2
            elif bmi > 25:
                risk_score += 1
            
            # Disease factor
            high_risk_diseases = ['hypertension', 'diabetes', 'heart disease', 'kidney', 'liver', 'cancer']
            medium_risk_diseases = ['asthma', 'arthritis', 'depression', 'anxiety', 'migraine']
            
            if any(disease_term in disease for disease_term in high_risk_diseases):
                risk_score += 3
            elif any(disease_term in disease for disease_term in medium_risk_diseases):
                risk_score += 2
            elif disease not in ['none', 'healthy', '']:
                risk_score += 1
            
            # Visit frequency factor
            if visit_freq > 0.5:
                risk_score += 2
            elif visit_freq > 0.2:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 7:
                risk_level = 'High'
                needs_treatment = 'Immediate'
            elif risk_score >= 4:
                risk_level = 'Medium'
                needs_treatment = 'Monitoring'
            else:
                risk_level = 'Low'
                needs_treatment = 'Routine'
            
            risk_labels.append(risk_level)
            treatment_labels.append(needs_treatment)
        
        return risk_labels, treatment_labels
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        feature_df = df[['age', 'height', 'weight', 'bmi', 'gender', 'blood_group', 
                        'disease', 'total_visits', 'visit_frequency']].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna({
            'age': feature_df['age'].median(),
            'height': feature_df['height'].median(),
            'weight': feature_df['weight'].median(),
            'bmi': feature_df['bmi'].median(),
            'gender': 'unknown',
            'blood_group': 'O+',
            'disease': 'None',
            'total_visits': 0,
            'visit_frequency': 0
        })
        
        # Encode categorical variables
        categorical_columns = ['gender', 'blood_group', 'disease']
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                feature_df[column] = self.label_encoders[column].fit_transform(feature_df[column].astype(str))
            else:
                le = self.label_encoders[column]
                unique_values = set(feature_df[column].astype(str))
                known_values = set(le.classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    le.classes_ = np.append(le.classes_, list(new_values))
                
                feature_df[column] = feature_df[column].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        return feature_df
    
    def train_models(self, df):
        """Train risk assessment and treatment recommendation models"""
        # Create BMI and visit frequency features
        df['bmi'] = df['weight'] / (df['height'] ** 2)
        df['total_visits'] = df['visits'].apply(len)
        df['visit_frequency'] = df['total_visits'] / df['age']
        
        # Create labels
        risk_labels, treatment_labels = self.create_risk_labels(df)
        
        # Prepare features
        X = self.prepare_features(df)
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target variables
        risk_encoder = LabelEncoder()
        treatment_encoder = LabelEncoder()
        
        y_risk = risk_encoder.fit_transform(risk_labels)
        y_treatment = treatment_encoder.fit_transform(treatment_labels)
        
        # Train models
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.treatment_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        
        self.risk_model.fit(X_scaled, y_risk)
        self.treatment_model.fit(X_scaled, y_treatment)
        
        # Store encoders
        self.risk_encoder = risk_encoder
        self.treatment_encoder = treatment_encoder
        
        return True
    
    def predict_patient_risk(self, patient_data):
        """Predict risk and treatment for a single patient"""
        if self.risk_model is None or self.treatment_model is None:
            return None
        
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])
        
        # Calculate features
        patient_df['bmi'] = patient_df['weight'] / (patient_df['height'] ** 2)
        visits = patient_data.get('visits', [])
        patient_df['total_visits'] = len(visits)
        patient_df['visit_frequency'] = len(visits) / max(1, patient_data.get('age', 1))
        
        # Prepare features
        X = self.prepare_features(patient_df)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        risk_pred = self.risk_model.predict(X_scaled)[0]
        treatment_pred = self.treatment_model.predict(X_scaled)[0]
        
        # Get probabilities
        risk_proba = self.risk_model.predict_proba(X_scaled)[0]
        treatment_proba = self.treatment_model.predict_proba(X_scaled)[0]
        
        # Decode predictions
        risk_level = self.risk_encoder.inverse_transform([risk_pred])[0]
        treatment_needed = self.treatment_encoder.inverse_transform([treatment_pred])[0]
        
        return {
            'risk_level': risk_level,
            'treatment_needed': treatment_needed,
            'risk_confidence': round(max(risk_proba), 3),
            'treatment_confidence': round(max(treatment_proba), 3),
            'risk_probabilities': {
                class_name: round(prob, 3) 
                for class_name, prob in zip(self.risk_encoder.classes_, risk_proba)
            }
        }
    
    def save_models(self, model_path='patient_risk_models.pkl'):
        """Save trained models"""
        model_data = {
            'risk_model': self.risk_model,
            'treatment_model': self.treatment_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'risk_encoder': self.risk_encoder,
            'treatment_encoder': self.treatment_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, model_path)
        return True
    
    def load_models(self, model_path='patient_risk_models.pkl'):
        """Load trained models"""
        try:
            model_data = joblib.load(model_path)
            self.risk_model = model_data['risk_model']
            self.treatment_model = model_data['treatment_model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.risk_encoder = model_data['risk_encoder']
            self.treatment_encoder = model_data['treatment_encoder']
            self.feature_columns = model_data['feature_columns']
            return True
        except:
            return False

DATA_FILE = 'patients.json'

def load_data() -> Dict[str, dict]:
    if not os.path.exists(DATA_FILE):
        save_data({})
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    return data

def save_data(data: Dict[str, dict]):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def initialize_ml_system():
    """Initialize ML system and try to load existing models"""
    global ml_system
    ml_system = PatientRiskAssessment()
    
    # Try to load existing models
    if os.path.exists('patient_risk_models.pkl'):
        success = ml_system.load_models()
        if success:
            print("✅ ML models loaded successfully")
        else:
            print("⚠️ Failed to load existing models")
    else:
        print("ℹ️ No existing models found")

def convert_json_to_dataframe():
    """Convert JSON patient data to DataFrame"""
    data = load_data()
    if not data:
        return None
    
    df_list = []
    for patient_id, patient_data in data.items():
        patient_record = patient_data.copy()
        patient_record['patient_id'] = patient_id
        df_list.append(patient_record)
    
    return pd.DataFrame(df_list)

def generate_patient_data(patient_id: str) -> dict:
    """Generate realistic patient data using Faker"""
    gender_choice = random.choice(['male', 'female', 'others'])
    
    if gender_choice == 'male':
        name = fake.name_male()
    elif gender_choice == 'female':
        name = fake.name_female()
    else:
        name = fake.name()
    
    age = random.randint(1, 100)
    
    if age < 18:
        height = random.uniform(0.5, 1.7)
        weight = random.uniform(10, 70)
    elif gender_choice == 'male':
        height = random.uniform(1.60, 1.95)
        weight = random.uniform(50, 120)
    else:
        height = random.uniform(1.50, 1.85)
        weight = random.uniform(40, 100)
    
    if age < 18:
        disease = random.choice(["Asthma", "Allergies", "Common Cold", "Healthy", "None"])
    elif age < 40:
        disease = random.choice(["Allergies", "Asthma", "Migraine", "Back Pain", "Healthy", "None", "Anxiety"])
    elif age < 60:
        disease = random.choice(["Hypertension", "Diabetes Type 2", "High Cholesterol", "Back Pain", "Arthritis", "Migraine"])
    else:
        disease = random.choice(["Hypertension", "Diabetes Type 2", "Heart Disease", "Arthritis", "Osteoporosis", "High Cholesterol"])
    
    return {
        'name': name,
        'city': fake.city(),
        'age': age,
        'gender': gender_choice,
        'height': round(height, 2),
        'weight': round(weight, 1),
        'phone': fake.phone_number(),
        'email': fake.email(),
        'blood_group': random.choice(BLOOD_GROUPS),
        'disease': disease,
        'visits': []
    }

async def generate_bulk_patients_async(count: int, start_id: int = 1, batch_size: int = 1000) -> Dict[str, dict]:
    """Generate bulk patient data asynchronously"""
    all_patients = {}
    
    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        batch_count = batch_end - batch_start
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            patients_batch = await loop.run_in_executor(
                executor, generate_bulk_patients, batch_count, start_id + batch_start
            )
        
        all_patients.update(patients_batch)
        
        if len(all_patients) % 10000 == 0:
            await asyncio.sleep(0.1)
    
    return all_patients

def generate_bulk_patients(count: int, start_id: int = 1) -> Dict[str, dict]:
    """Generate bulk patient data efficiently"""
    patients = {}
    
    for i in range(count):
        patient_id = f"P{start_id + i:06d}"
        patients[patient_id] = generate_patient_data(patient_id)
    
    return patients

# Initialize ML system on startup
initialize_ml_system()

@app.get("/")
def hello():
    return {'message': 'Enhanced Patient Management System API with ML Risk Assessment'}

@app.get('/about')
def about():
    return {
        'message': 'A fully functional API to manage patient records with ML-powered risk assessment',
        'features': [
            'CRUD operations for patients',
            'Bulk patient data generation using Faker',
            'Machine Learning risk assessment',
            'Treatment recommendation system',
            'Support for up to 100,000 patients',
            'JSON to CSV conversion',
            'BMI calculation and health verdict',
            'Visit tracking and free service eligibility'
        ]
    }

@app.get('/stats')
def get_stats():
    """Get database statistics"""
    data = load_data()
    total_patients = len(data)
    
    if total_patients == 0:
        return {
            'total_patients': 0,
            'message': 'No patients in database'
        }
    
    ages = [p.get('age', 0) for p in data.values()]
    genders = [p.get('gender', '') for p in data.values()]
    blood_groups = [p.get('blood_group', '') for p in data.values()]
    diseases = [p.get('disease', '') for p in data.values()]
    
    from collections import Counter
    
    return {
        'total_patients': total_patients,
        'average_age': round(sum(ages) / len(ages), 1),
        'age_range': {'min': min(ages), 'max': max(ages)},
        'gender_distribution': dict(Counter(genders)),
        'blood_group_distribution': dict(Counter(blood_groups)),
        'top_diseases': dict(Counter(diseases).most_common(10)),
        'ml_model_status': 'Trained' if ml_system and ml_system.risk_model else 'Not Trained'
    }

@app.post('/generate')
async def generate_patients(request: BulkGenerateRequest):
    """Generate bulk patient data using Faker"""
    start_time = time.time()
    
    try:
        existing_data = {} if request.clear_existing else load_data()
        existing_count = len(existing_data)
        
        if existing_data:
            max_id = 0
            for patient_id in existing_data.keys():
                if patient_id.startswith('P'):
                    try:
                        id_num = int(patient_id[1:])
                        max_id = max(max_id, id_num)
                    except ValueError:
                        continue
            start_id = max_id + 1
        else:
            start_id = 1
        
        new_patients = await generate_bulk_patients_async(request.count, start_id)
        
        if not request.clear_existing:
            new_patients.update(existing_data)
        
        save_data(new_patients)
        
        end_time = time.time()
        generation_time = round(end_time - start_time, 2)
        
        return {
            'message': f'Successfully generated {request.count} patients',
            'total_patients': len(new_patients),
            'new_patients_added': request.count,
            'existing_patients': existing_count if not request.clear_existing else 0,
            'generation_time_seconds': generation_time,
            'cleared_existing': request.clear_existing,
            'id_range': f'P{start_id:06d} to P{start_id + request.count - 1:06d}',
            'ml_training_recommended': len(new_patients) >= 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating patients: {str(e)}')

@app.post('/ml/train')
async def train_ml_models(background_tasks: BackgroundTasks, request: MLTrainingRequest = None):
    """Train machine learning models for risk assessment"""
    global ml_system
    
    data = load_data()
    if len(data) < 50:
        raise HTTPException(status_code=400, detail='Need at least 50 patients to train ML models')
    
    try:
        if ml_system is None:
            ml_system = PatientRiskAssessment()
        
        # Convert to DataFrame
        df = convert_json_to_dataframe()
        if df is None:
            raise HTTPException(status_code=400, detail='No patient data found')
        
        # Train models
        success = ml_system.train_models(df)
        
        if success:
            # Save models
            ml_system.save_models()
            
            return {
                'message': 'ML models trained successfully',
                'patients_used': len(df),
                'model_status': 'Ready for predictions',
                'features_used': ml_system.feature_columns
            }
        else:
            raise HTTPException(status_code=500, detail='Failed to train ML models')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error training ML models: {str(e)}')

@app.get('/ml/status')
def get_ml_status():
    """Get ML model status"""
    if ml_system is None:
        return {'status': 'Not initialized', 'models_trained': False}
    
    return {
        'status': 'Initialized',
        'models_trained': ml_system.risk_model is not None,
        'feature_columns': ml_system.feature_columns if ml_system.feature_columns else [],
        'models_available': {
            'risk_assessment': ml_system.risk_model is not None,
            'treatment_recommendation': ml_system.treatment_model is not None
        }
    }

@app.get('/patient/{patient_id}/risk')
def assess_patient_risk(patient_id: str = Path(..., description='ID of the patient')):
    """Get ML-powered risk assessment for a patient"""
    if ml_system is None or ml_system.risk_model is None:
        raise HTTPException(status_code=400, detail='ML models not trained. Please train models first.')
    
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    
    patient_data = data[patient_id]
    patient_data['patient_id'] = patient_id
    
    try:
        prediction = ml_system.predict_patient_risk(patient_data)
        
        if prediction is None:
            raise HTTPException(status_code=500, detail='Failed to generate risk prediction')
        
        # Get basic patient info
        patient = Patient(id=patient_id, **patient_data)
        
        return {
            'patient_id': patient_id,
            'patient_name': patient.name,
            'basic_info': {
                'age': patient.age,
                'bmi': patient.bmi,
                'bmi_category': patient.verdict,
                'primary_disease': patient.disease,
                'total_visits': len(patient.visits)
            },
            'risk_assessment': prediction,
            'recommendations': generate_recommendations(prediction['risk_level'], prediction['treatment_needed'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating risk assessment: {str(e)}')

def generate_recommendations(risk_level: str, treatment_needed: str) -> dict:
    """Generate recommendations based on risk assessment"""
    recommendations = {
        'immediate_actions': [],
        'follow_up': [],
        'lifestyle': [],
        'monitoring': []
    }
    
    if risk_level == 'High':
        recommendations['immediate_actions'] = [
            'Schedule immediate consultation with specialist',
            'Consider emergency intervention if symptoms worsen',
            'Review all current medications'
        ]
        recommendations['follow_up'] = ['Weekly follow-up appointments', 'Specialist referral within 48 hours']
        recommendations['monitoring'] = ['Daily vital signs monitoring', 'Symptom tracking']
        
    elif risk_level == 'Medium':
        recommendations['immediate_actions'] = ['Schedule appointment within 1-2 weeks']
        recommendations['follow_up'] = ['Bi-weekly monitoring', 'Reassess in 1 month']
        recommendations['lifestyle'] = ['Diet consultation', 'Exercise program evaluation']
        recommendations['monitoring'] = ['Weekly health checks', 'Track key symptoms']
        
    else:  # Low risk
        recommendations['follow_up'] = ['Routine annual check-up', 'Continue current care plan']
        recommendations['lifestyle'] = ['Maintain healthy lifestyle', 'Regular exercise', 'Balanced diet']
        recommendations['monitoring'] = ['Monthly self-assessment', 'Annual comprehensive screening']
    
    return recommendations

@app.get('/ml/batch-assess')
async def batch_risk_assessment(limit: int = Query(100, le=1000, description='Max patients to assess')):
    """Perform batch risk assessment on multiple patients"""
    if ml_system is None or ml_system.risk_model is None:
        raise HTTPException(status_code=400, detail='ML models not trained')
    
    data = load_data()
    if not data:
        raise HTTPException(status_code=404, detail='No patients found')
    
    try:
        # Convert to DataFrame
        df = convert_json_to_dataframe()
        
        # Limit the number of patients
        if len(df) > limit:
            df = df.head(limit)
        
        # Perform batch predictions
        results = []
        for _, patient_row in df.iterrows():
            patient_data = patient_row.to_dict()
            prediction = ml_system.predict_patient_risk(patient_data)
            
            if prediction:
                results.append({
                    'patient_id': patient_data['patient_id'],
                    'name': patient_data.get('name', 'Unknown'),
                    'age': patient_data.get('age', 0),
                    'risk_level': prediction['risk_level'],
                    'treatment_needed': prediction['treatment_needed'],
                    'risk_confidence': prediction['risk_confidence']
                })
        
        # Calculate summary statistics
        risk_summary = {}
        treatment_summary = {}
        
        for result in results:
            risk = result['risk_level']
            treatment = result['treatment_needed']
            
            risk_summary[risk] = risk_summary.get(risk, 0) + 1
            treatment_summary[treatment] = treatment_summary.get(treatment, 0) + 1
        
        return {
            'total_assessed': len(results),
            'risk_distribution': risk_summary,
            'treatment_distribution': treatment_summary,
            'high_risk_patients': [r for r in results if r['risk_level'] == 'High'],
            'all_results': results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error in batch assessment: {str(e)}')

@app.get('/export/csv')
def export_patients_csv(include_predictions: bool = Query(False, description='Include ML predictions')):
    """Export patient data to CSV format"""
    data = load_data()
    if not data:
        raise HTTPException(status_code=404, detail='No patients found')
    
    try:
        df = convert_json_to_dataframe()
        
        # Add calculated fields
        df['bmi'] = df['weight'] / (df['height'] ** 2)
        df['total_visits'] = df['visits'].apply(len)
        
        # Add BMI category
        def get_bmi_category(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        df['bmi_category'] = df['bmi'].apply(get_bmi_category)
        
        # Add ML predictions if requested and models are available
        if include_predictions and ml_system and ml_system.risk_model:
            predictions = []
            for _, patient_row in df.iterrows():
                patient_data = patient_row.to_dict()
                prediction = ml_system.predict_patient_risk(patient_data)
                
                if prediction:
                    predictions.append({
                        'predicted_risk_level': prediction['risk_level'],
                        'predicted_treatment': prediction['treatment_needed'],
                        'risk_confidence': prediction['risk_confidence']
                    })
                else:
                    predictions.append({
                        'predicted_risk_level': 'Unknown',
                        'predicted_treatment': 'Unknown',
                        'risk_confidence': 0.0
                    })
            
            pred_df = pd.DataFrame(predictions)
            df = pd.concat([df, pred_df], axis=1)
        
        # Convert visits list to string for CSV compatibility
        df['visits'] = df['visits'].apply(lambda x: ';'.join(x) if x else '')
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'patients_export_{timestamp}.csv'
        csv_path = filename
        
        df.to_csv(csv_path, index=False)
        
        return FileResponse(
            path=csv_path,
            filename=filename,
            media_type='text/csv',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error exporting CSV: {str(e)}')

# Include all the original endpoints
@app.get('/view')
def view():
    data = load_data()
    return {pid: Patient(id=pid, **pdata).as_dict() for pid, pdata in data.items()}

@app.get('/view/paginated')
def view_paginated(
    page: int = Query(1, ge=1, description='Page number'),
    limit: int = Query(50, ge=1, le=1000, description='Number of patients per page')
):
    """Get paginated patient data for large datasets"""
    data = load_data()
    total_patients = len(data)
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    patient_ids = list(data.keys())
    paginated_ids = patient_ids[start_idx:end_idx]
    
    patients = {}
    for pid in paginated_ids:
        patients[pid] = Patient(id=pid, **data[pid]).as_dict()
    
    return {
        'patients': patients,
        'pagination': {
            'current_page': page,
            'per_page': limit,
            'total_patients': total_patients,
            'total_pages': (total_patients + limit - 1) // limit,
            'has_next': end_idx < total_patients,
            'has_prev': page > 1
        }
    }

@app.get('/patient/{patient_id}')
def view_patient(patient_id: str = Path(..., description='ID of the patient')):
    data = load_data()
    if patient_id in data:
        patient_data = data[patient_id]
        patient = Patient(id=patient_id, **patient_data)
        visits = patient.visits
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        week_visits = [
            v for v in visits
            if datetime.fromisoformat(v) >= week_start
        ]
        free_service = len(week_visits) > 2
        now = datetime.now()
        month_visits = [
            v for v in visits
            if datetime.fromisoformat(v).year == now.year and datetime.fromisoformat(v).month == now.month
        ]
        out = patient.as_dict()
        out['free_service'] = free_service
        out['monthly_visits'] = len(month_visits)
        
        # Add ML risk assessment if available
        if ml_system and ml_system.risk_model:
            try:
                risk_assessment = ml_system.predict_patient_risk(patient_data)
                out['ml_risk_assessment'] = risk_assessment
            except:
                out['ml_risk_assessment'] = None
        
        return out
    raise HTTPException(status_code=404, detail='Patient not found')

@app.post('/create')
def create_patient(patient: Patient):
    data = load_data()
    if patient.id in data:
        raise HTTPException(status_code=400, detail='Patient already exists')
    patient_dict = patient.dict()
    data[patient.id] = {key: patient_dict[key] for key in patient_dict if key != 'id'}
    save_data(data)
    return JSONResponse(status_code=201, content={'message': 'patient created successfully'})

@app.post('/visit/{patient_id}')
def record_visit(patient_id: str):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    now = datetime.now().isoformat()
    visits = data[patient_id].get('visits', [])
    visits.append(now)
    data[patient_id]['visits'] = visits
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_visits = [
        v for v in visits
        if datetime.fromisoformat(v) >= week_start
    ]
    free_service = len(week_visits) > 2
    save_data(data)
    return {
        'message': 'Visit recorded',
        'free_service': free_service
    }

@app.put('/edit/{patient_id}')
def update_patient(patient_id: str, patient_update: PatientUpdate):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    existing_patient_info = data[patient_id]
    updated_info = patient_update.dict(exclude_unset=True)
    existing_patient_info.update(updated_info)
    patient_obj = Patient(id=patient_id, **existing_patient_info)
    data[patient_id] = patient_obj.dict(exclude={'id'})
    save_data(data)
    return JSONResponse(status_code=200, content={'message': 'patient updated'})

@app.delete('/delete/{patient_id}')
def delete_patient(patient_id: str):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    del data[patient_id]
    save_data(data)
    return JSONResponse(status_code=200, content={'message': 'patient deleted'})

@app.delete('/clear')
def clear_all_patients():
    """Clear all patient data"""
    save_data({})
    return {'message': 'All patient data cleared successfully'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

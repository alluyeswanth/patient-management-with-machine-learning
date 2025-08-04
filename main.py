from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Optional, List, Dict
import json
from datetime import datetime, timedelta
import os
from faker import Faker
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()
fake = Faker()

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

# Risk Assessment Model
class HealthRiskModel:
    def __init__(self):
        self.risk_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        self.treatment_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        self.gender_encoder = LabelEncoder()
        self.disease_encoder = LabelEncoder()
        self.blood_encoder = LabelEncoder()
        self.is_trained = False
        
    def _generate_training_data(self, n_samples=5000):
        """Generate synthetic training data for the ML models"""
        data = []
        risk_labels = []
        treatment_labels = []
        
        for _ in range(n_samples):
            age = random.randint(1, 100)
            gender = random.choice(['male', 'female', 'others'])
            
            # Generate realistic height/weight based on age and gender
            if age < 18:
                height = random.uniform(0.5, 1.7)
                weight = random.uniform(10, 70)
            elif gender == 'male':
                height = random.uniform(1.60, 1.95)
                weight = random.uniform(50, 120)
            else:
                height = random.uniform(1.50, 1.85)
                weight = random.uniform(40, 100)
            
            bmi = weight / (height ** 2)
            disease = random.choice(DISEASES)
            blood_group = random.choice(BLOOD_GROUPS)
            visit_count = random.randint(0, 20)
            
            # Simple risk assessment logic for training
            risk_score = 0
            
            # Age factor
            if age > 65:
                risk_score += 3
            elif age > 50:
                risk_score += 2
            elif age < 18:
                risk_score += 1
            
            # BMI factor
            if bmi > 30 or bmi < 18.5:
                risk_score += 2
            elif bmi > 25:
                risk_score += 1
            
            # Disease factor
            high_risk_diseases = ["Heart Disease", "Diabetes Type 2", "Hypertension", 
                                "Liver Disease", "Kidney Stones", "Tuberculosis", "Hepatitis"]
            moderate_risk_diseases = ["Asthma", "High Cholesterol", "Thyroid Disorder", 
                                    "Osteoporosis", "Depression", "Arthritis"]
            
            if disease in high_risk_diseases:
                risk_score += 3
            elif disease in moderate_risk_diseases:
                risk_score += 2
            elif disease in ["None", "Healthy"]:
                risk_score += 0
            else:
                risk_score += 1
            
            # Visit frequency factor
            if visit_count > 10:
                risk_score += 2
            elif visit_count > 5:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 7:
                risk_level = "High"
            elif risk_score >= 4:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            # Treatment prediction (simplified logic)
            needs_treatment = (risk_score >= 5 or 
                             disease in high_risk_diseases or 
                             bmi > 35 or bmi < 16 or 
                             age > 70)
            
            data.append([age, height, weight, bmi, visit_count, gender, disease, blood_group])
            risk_labels.append(risk_level)
            treatment_labels.append(1 if needs_treatment else 0)
        
        return data, risk_labels, treatment_labels
    
    def train(self):
        """Train the ML models"""
        # Generate training data
        data, risk_labels, treatment_labels = self._generate_training_data()
        
        # Prepare encoders
        genders = [row[5] for row in data]
        diseases = [row[6] for row in data]
        blood_groups = [row[7] for row in data]
        
        self.gender_encoder.fit(genders + ['male', 'female', 'others'])
        self.disease_encoder.fit(diseases + DISEASES)
        self.blood_encoder.fit(blood_groups + BLOOD_GROUPS)
        
        # Encode categorical features
        X = []
        for row in data:
            encoded_row = [
                row[0],  # age
                row[1],  # height
                row[2],  # weight
                row[3],  # bmi
                row[4],  # visit_count
                self.gender_encoder.transform([row[5]])[0],
                self.disease_encoder.transform([row[6]])[0],
                self.blood_encoder.transform([row[7]])[0]
            ]
            X.append(encoded_row)
        
        X = np.array(X)
        
        # Train models
        self.risk_model.fit(X, risk_labels)
        self.treatment_model.fit(X, treatment_labels)
        self.is_trained = True
        print("ML models trained successfully!")
    
    def predict_risk_and_treatment(self, patient_data):
        """Predict risk level and treatment need for a patient"""
        if not self.is_trained:
            self.train()
        
        # Extract features
        age = patient_data.get('age', 25)
        height = patient_data.get('height', 1.7)
        weight = patient_data.get('weight', 70)
        bmi = weight / (height ** 2)
        visit_count = len(patient_data.get('visits', []))
        gender = patient_data.get('gender', 'male')
        disease = patient_data.get('disease', 'None')
        blood_group = patient_data.get('blood_group', 'O+')
        
        # Handle unknown categories
        try:
            gender_encoded = self.gender_encoder.transform([gender])[0]
        except:
            gender_encoded = self.gender_encoder.transform(['male'])[0]
        
        try:
            disease_encoded = self.disease_encoder.transform([disease])[0]
        except:
            disease_encoded = self.disease_encoder.transform(['None'])[0]
        
        try:
            blood_encoded = self.blood_encoder.transform([blood_group])[0]
        except:
            blood_encoded = self.blood_encoder.transform(['O+'])[0]
        
        # Prepare input
        X = np.array([[age, height, weight, bmi, visit_count, 
                      gender_encoded, disease_encoded, blood_encoded]])
        
        # Make predictions
        risk_prediction = self.risk_model.predict(X)[0]
        risk_probability = self.risk_model.predict_proba(X)[0]
        
        treatment_prediction = self.treatment_model.predict(X)[0]
        treatment_probability = self.treatment_model.predict_proba(X)[0]
        
        # Get risk probabilities
        risk_classes = self.risk_model.classes_
        risk_probs = {risk_classes[i]: float(prob) for i, prob in enumerate(risk_probability)}
        
        return {
            'risk_level': risk_prediction,
            'risk_probabilities': risk_probs,
            'needs_treatment': bool(treatment_prediction),
            'treatment_confidence': float(treatment_probability[treatment_prediction]),
            'risk_factors': self._analyze_risk_factors(patient_data, bmi),
            'recommendations': self._get_recommendations(risk_prediction, bool(treatment_prediction), patient_data)
        }
    
    def _analyze_risk_factors(self, patient_data, bmi):
        """Analyze specific risk factors for the patient"""
        factors = []
        age = patient_data.get('age', 25)
        disease = patient_data.get('disease', 'None')
        visit_count = len(patient_data.get('visits', []))
        
        if age > 65:
            factors.append("Advanced age (>65 years)")
        elif age > 50:
            factors.append("Middle age (50-65 years)")
        
        if bmi > 30:
            factors.append("Obesity (BMI > 30)")
        elif bmi > 25:
            factors.append("Overweight (BMI > 25)")
        elif bmi < 18.5:
            factors.append("Underweight (BMI < 18.5)")
        
        high_risk_diseases = ["Heart Disease", "Diabetes Type 2", "Hypertension", 
                            "Liver Disease", "Kidney Stones", "Tuberculosis", "Hepatitis"]
        if disease in high_risk_diseases:
            factors.append(f"High-risk condition: {disease}")
        
        if visit_count > 10:
            factors.append("Frequent medical visits (>10)")
        elif visit_count > 5:
            factors.append("Multiple medical visits (>5)")
        
        return factors if factors else ["No significant risk factors identified"]
    
    def _get_recommendations(self, risk_level, needs_treatment, patient_data):
        """Generate health recommendations based on predictions"""
        recommendations = []
        age = patient_data.get('age', 25)
        disease = patient_data.get('disease', 'None')
        bmi = patient_data.get('weight', 70) / (patient_data.get('height', 1.7) ** 2)
        
        if risk_level == "High":
            recommendations.append("Immediate medical consultation recommended")
            recommendations.append("Regular health monitoring required")
            recommendations.append("Consider lifestyle modifications")
        elif risk_level == "Moderate":
            recommendations.append("Schedule regular check-ups")
            recommendations.append("Monitor symptoms closely")
        else:
            recommendations.append("Maintain healthy lifestyle")
            recommendations.append("Annual health screening")
        
        if needs_treatment:
            recommendations.append("Medical treatment may be required")
            recommendations.append("Consult with healthcare provider")
        
        if bmi > 25:
            recommendations.append("Consider weight management program")
        elif bmi < 18.5:
            recommendations.append("Nutritional counseling recommended")
        
        if age > 65:
            recommendations.append("Senior health screening protocols")
        
        return recommendations

# Initialize ML model
ml_model = HealthRiskModel()

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
        
        # Add ML predictions
        try:
            ml_predictions = ml_model.predict_risk_and_treatment(d)
            d.update(ml_predictions)
        except Exception as e:
            # Fallback if ML prediction fails
            d['risk_level'] = 'Unknown'
            d['needs_treatment'] = False
            d['risk_factors'] = ['ML prediction unavailable']
            d['recommendations'] = ['Consult healthcare provider']
        
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

def generate_patient_data(patient_id: str) -> dict:
    """Generate realistic patient data using Faker"""
    gender_choice = random.choice(['male', 'female', 'others'])
    
    # Generate name based on gender
    if gender_choice == 'male':
        name = fake.name_male()
    elif gender_choice == 'female':
        name = fake.name_female()
    else:
        name = fake.name()
    
    # Generate realistic height and weight based on age and gender
    age = random.randint(1, 100)
    
    if age < 18:  # Children
        height = random.uniform(0.5, 1.7)
        weight = random.uniform(10, 70)
    elif gender_choice == 'male':
        height = random.uniform(1.60, 1.95)
        weight = random.uniform(50, 120)
    else:  # female or others
        height = random.uniform(1.50, 1.85)
        weight = random.uniform(40, 100)
    
    # Generate disease based on age
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

def generate_bulk_patients(count: int, start_id: int = 1) -> Dict[str, dict]:
    """Generate bulk patient data efficiently"""
    patients = {}
    
    for i in range(count):
        patient_id = f"P{start_id + i:06d}"  # Format: P000001, P000002, etc.
        patients[patient_id] = generate_patient_data(patient_id)
    
    return patients

async def generate_bulk_patients_async(count: int, start_id: int = 1, batch_size: int = 1000) -> Dict[str, dict]:
    """Generate bulk patient data asynchronously for better performance"""
    all_patients = {}
    
    # Process in batches to avoid memory issues
    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        batch_count = batch_end - batch_start
        
        # Use thread pool for CPU-intensive task
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            patients_batch = await loop.run_in_executor(
                executor, generate_bulk_patients, batch_count, start_id + batch_start
            )
        
        all_patients.update(patients_batch)
        
        # Small delay to prevent overwhelming the system
        if len(all_patients) % 10000 == 0:
            await asyncio.sleep(0.1)
    
    return all_patients

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
            'Machine Learning risk assessment (High/Moderate/Low)',
            'Treatment need prediction',
            'Risk factor analysis',
            'Health recommendations',
            'BMI calculation and health verdict',
            'Visit tracking and free service eligibility'
        ],
        'ml_models': {
            'risk_assessment': 'Random Forest Classifier for risk level prediction',
            'treatment_prediction': 'Random Forest Classifier for treatment need prediction',
            'features_used': ['age', 'height', 'weight', 'bmi', 'visit_count', 'gender', 'disease', 'blood_group']
        }
    }

@app.get('/stats')
def get_stats():
    """Get database statistics including ML insights"""
    data = load_data()
    total_patients = len(data)
    
    if total_patients == 0:
        return {
            'total_patients': 0,
            'message': 'No patients in database'
        }
    
    # Calculate statistics
    ages = [p.get('age', 0) for p in data.values()]
    genders = [p.get('gender', '') for p in data.values()]
    blood_groups = [p.get('blood_group', '') for p in data.values()]
    diseases = [p.get('disease', '') for p in data.values()]
    
    # ML-based risk analysis
    risk_levels = []
    treatment_needs = []
    
    for pid, pdata in data.items():
        try:
            patient_dict = pdata.copy()
            patient_dict['id'] = pid
            ml_result = ml_model.predict_risk_and_treatment(patient_dict)
            risk_levels.append(ml_result['risk_level'])
            treatment_needs.append(ml_result['needs_treatment'])
        except:
            risk_levels.append('Unknown')
            treatment_needs.append(False)
    
    from collections import Counter
    
    return {
        'total_patients': total_patients,
        'average_age': round(sum(ages) / len(ages), 1),
        'age_range': {'min': min(ages), 'max': max(ages)},
        'gender_distribution': dict(Counter(genders)),
        'blood_group_distribution': dict(Counter(blood_groups)),
        'top_diseases': dict(Counter(diseases).most_common(10)),
        'ml_insights': {
            'risk_distribution': dict(Counter(risk_levels)),
            'treatment_needs': {
                'needs_treatment': sum(treatment_needs),
                'no_treatment_needed': len(treatment_needs) - sum(treatment_needs),
                'treatment_rate': round((sum(treatment_needs) / len(treatment_needs)) * 100, 1) if treatment_needs else 0
            }
        }
    }

@app.post('/generate')
async def generate_patients(request: BulkGenerateRequest):
    """Generate bulk patient data using Faker"""
    start_time = time.time()
    
    try:
        # Load existing data
        existing_data = {} if request.clear_existing else load_data()
        existing_count = len(existing_data)
        
        # Determine starting ID
        if existing_data:
            # Find the highest existing ID number
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
        
        # Generate new patients
        new_patients = await generate_bulk_patients_async(request.count, start_id)
        
        # Merge with existing data
        if not request.clear_existing:
            new_patients.update(existing_data)
        
        # Save to file
        save_data(new_patients)
        
        end_time = time.time()
        generation_time = round(end_time - start_time, 2)
        
        return {
            'message': f'Successfully generated {request.count} patients with ML risk assessment',
            'total_patients': len(new_patients),
            'new_patients_added': request.count,
            'existing_patients': existing_count if not request.clear_existing else 0,
            'generation_time_seconds': generation_time,
            'cleared_existing': request.clear_existing,
            'id_range': f'P{start_id:06d} to P{start_id + request.count - 1:06d}',
            'ml_features': 'Risk assessment and treatment prediction enabled'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating patients: {str(e)}')

@app.get('/view')
def view():
    data = load_data()
    return {pid: Patient(id=pid, **pdata).as_dict() for pid, pdata in data.items()}

@app.get('/view/paginated')
def view_paginated(
    page: int = Query(1, ge=1, description='Page number'),
    limit: int = Query(50, ge=1, le=1000, description='Number of patients per page')
):
    """Get paginated patient data for large datasets with ML predictions"""
    data = load_data()
    total_patients = len(data)
    
    # Calculate pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    # Get patient IDs and slice
    patient_ids = list(data.keys())
    paginated_ids = patient_ids[start_idx:end_idx]
    
    # Build response
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
def view_patient(patient_id: str = Path(..., description='ID of the patient in the DB', example='P000001')):
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
        return out
    raise HTTPException(status_code=404, detail='Patient not found')

@app.get('/risk-analysis/{patient_id}')
def get_detailed_risk_analysis(patient_id: str = Path(..., description='ID of the patient')):
    """Get detailed ML-based risk analysis for a specific patient"""
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    
    patient_data = data[patient_id]
    patient_data['id'] = patient_id
    
    try:
        ml_result = ml_model.predict_risk_and_treatment(patient_data)
        
        # Add additional analysis
        bmi = patient_data['weight'] / (patient_data['height'] ** 2)
        age = patient_data['age']
        
        detailed_analysis = {
            'patient_id': patient_id,
            'patient_name': patient_data.get('name', 'Unknown'),
            'risk_assessment': ml_result,
            'health_metrics': {
                'bmi': round(bmi, 2),
                'bmi_category': 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese',
                'age_category': 'Child' if age < 18 else 'Young Adult' if age < 30 else 'Adult' if age < 60 else 'Senior'
            },
            'visit_history': {
                'total_visits': len(patient_data.get('visits', [])),
                'recent_visits': len([v for v in patient_data.get('visits', []) 
                                    if (datetime.now() - datetime.fromisoformat(v)).days <= 30])
            }
        }
        
        return detailed_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error analyzing patient risk: {str(e)}')

@app.get('/population-risk-analysis')
def get_population_risk_analysis():
    """Get ML-based risk analysis for the entire patient population"""
    data = load_data()
    if not data:
        return {'message': 'No patients in database'}
    
    risk_summary = {
        'total_patients': len(data),
        'risk_distribution': {'High': 0, 'Moderate': 0, 'Low': 0, 'Unknown': 0},
        'treatment_needs': {'needs_treatment': 0, 'no_treatment': 0},
        'risk_by_age_group': {},
        'risk_by_gender': {},
        'common_risk_factors': {}
    }
    
    all_risk_factors = []
    
    for pid, pdata in data.items():
        try:
            patient_dict = pdata.copy()
            patient_dict['id'] = pid
            ml_result = ml_model.predict_risk_and_treatment(patient_dict)
            
            # Update risk distribution
            risk_level = ml_result['risk_level']
            risk_summary['risk_distribution'][risk_level] += 1
            
            # Update treatment needs
            if ml_result['needs_treatment']:
                risk_summary['treatment_needs']['needs_treatment'] += 1
            else:
                risk_summary['treatment_needs']['no_treatment'] += 1
            
            # Risk by age group
            age = pdata.get('age', 0)
            age_group = 'Child' if age < 18 else 'Young Adult' if age < 30 else 'Adult' if age < 60 else 'Senior'
            if age_group not in risk_summary['risk_by_age_group']:
                risk_summary['risk_by_age_group'][age_group] = {'High': 0, 'Moderate': 0, 'Low': 0}
            risk_summary['risk_by_age_group'][age_group][risk_level] += 1
            
            # Risk by gender
            gender = pdata.get('gender', 'unknown')
            if gender not in risk_summary['risk_by_gender']:
                risk_summary['risk_by_gender'][gender] = {'High': 0, 'Moderate': 0, 'Low': 0}
            risk_summary['risk_by_gender'][gender][risk_level] += 1
            
            # Collect risk factors
            all_risk_factors.extend(ml_result.get('risk_factors', []))
            
        except Exception:
            risk_summary['risk_distribution']['Unknown'] += 1
    
    # Count common risk factors
    from collections import Counter
    risk_factor_counts = Counter(all_risk_factors)
    risk_summary['common_risk_factors'] = dict(risk_factor_counts.most_common(10))
    
    # Calculate percentages
    total = risk_summary['total_patients']
    risk_summary['risk_percentages'] = {
        level: round((count / total) * 100, 1)
        for level, count in risk_summary['risk_distribution'].items()
    }
    
    return risk_summary

@app.get('/sort')
def sort_patients(
    sort_by: str = Query(..., description='Sort field: height, weight, bmi, age, name, risk_level'),
    order: str = Query('asc', description='Sort order: asc or desc'),
    limit: int = Query(100, ge=1, le=10000, description='Maximum number of results')
):
    valid_fields = ['height', 'weight', 'bmi', 'age', 'name', 'risk_level']
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f'Invalid field, select from {valid_fields}')
    if order not in ['asc', 'desc']:
        raise HTTPException(status_code=400, detail='Invalid order, select between asc and desc')
    
    data = load_data()
    patients = [Patient(id=pid, **pdata) for pid, pdata in data.items()]
    reverse = (order == 'desc')
    
    if sort_by == "bmi":
        sorted_patients = sorted(patients, key=lambda x: x.bmi, reverse=reverse)
    elif sort_by == "risk_level":
        # Sort by risk level (High > Moderate > Low)
        risk_order = {'High': 3, 'Moderate': 2, 'Low': 1, 'Unknown': 0}
        sorted_patients = sorted(patients, key=lambda x: risk_order.get(x.as_dict().get('risk_level', 'Unknown'), 0), reverse=reverse)
    else:
        sorted_patients = sorted(patients, key=lambda x: getattr(x, sort_by), reverse=reverse)
    
    # Apply limit
    limited_patients = sorted_patients[:limit]
    
    return [p.as_dict() for p in limited_patients]

@app.get('/search')
def search_patients(
    query: str = Query(..., min_length=1, description='Search query'),
    field: str = Query('name', description='Field to search: name, city, disease, blood_group, risk_level'),
    limit: int = Query(100, ge=1, le=1000, description='Maximum results')
):
    """Search patients by various fields including ML predictions"""
    valid_fields = ['name', 'city', 'disease', 'blood_group', 'email', 'risk_level']
    if field not in valid_fields:
        raise HTTPException(status_code=400, detail=f'Invalid field, select from {valid_fields}')
    
    data = load_data()
    results = []
    
    for pid, pdata in data.items():
        if field == 'risk_level':
            # Special handling for ML-predicted risk level
            try:
                patient_dict = pdata.copy()
                patient_dict['id'] = pid
                ml_result = ml_model.predict_risk_and_treatment(patient_dict)
                field_value = ml_result['risk_level'].lower()
            except:
                field_value = 'unknown'
        else:
            field_value = pdata.get(field, '').lower()
        
        if query.lower() in field_value:
            patient = Patient(id=pid, **pdata)
            results.append(patient.as_dict())
            
            if len(results) >= limit:
                break
    
    return {
        'query': query,
        'field': field,
        'results_count': len(results),
        'patients': results
    }

@app.get('/high-risk-patients')
def get_high_risk_patients(limit: int = Query(100, ge=1, le=1000, description='Maximum results')):
    """Get all high-risk patients based on ML predictions"""
    data = load_data()
    high_risk_patients = []
    
    for pid, pdata in data.items():
        try:
            patient_dict = pdata.copy()
            patient_dict['id'] = pid
            ml_result = ml_model.predict_risk_and_treatment(patient_dict)
            
            if ml_result['risk_level'] == 'High':
                patient = Patient(id=pid, **pdata)
                patient_data = patient.as_dict()
                high_risk_patients.append(patient_data)
                
                if len(high_risk_patients) >= limit:
                    break
        except:
            continue
    
    return {
        'high_risk_count': len(high_risk_patients),
        'patients': high_risk_patients,
        'message': f'Found {len(high_risk_patients)} high-risk patients requiring immediate attention'
    }

@app.get('/treatment-needed')
def get_patients_needing_treatment(limit: int = Query(100, ge=1, le=1000, description='Maximum results')):
    """Get all patients who need treatment based on ML predictions"""
    data = load_data()
    treatment_needed = []
    
    for pid, pdata in data.items():
        try:
            patient_dict = pdata.copy()
            patient_dict['id'] = pid
            ml_result = ml_model.predict_risk_and_treatment(patient_dict)
            
            if ml_result['needs_treatment']:
                patient = Patient(id=pid, **pdata)
                patient_data = patient.as_dict()
                treatment_needed.append(patient_data)
                
                if len(treatment_needed) >= limit:
                    break
        except:
            continue
    
    return {
        'treatment_needed_count': len(treatment_needed),
        'patients': treatment_needed,
        'message': f'Found {len(treatment_needed)} patients requiring medical treatment'
    }

@app.post('/create')
def create_patient(patient: Patient):
    data = load_data()
    if patient.id in data:
        raise HTTPException(status_code=400, detail='Patient already exists')
    patient_dict = patient.dict()
    data[patient.id] = {key: patient_dict[key] for key in patient_dict if key != 'id'}
    save_data(data)
    return JSONResponse(status_code=201, content={'message': 'patient created successfully with ML risk assessment'})

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
    
    # Get updated ML predictions after visit
    try:
        patient_dict = data[patient_id].copy()
        patient_dict['id'] = patient_id
        ml_result = ml_model.predict_risk_and_treatment(patient_dict)
        risk_info = {
            'risk_level': ml_result['risk_level'],
            'needs_treatment': ml_result['needs_treatment']
        }
    except:
        risk_info = {'risk_level': 'Unknown', 'needs_treatment': False}
    
    save_data(data)
    return {
        'message': 'Visit recorded with updated risk assessment',
        'free_service': free_service,
        'updated_risk_assessment': risk_info
    }

@app.get('/visits/{patient_id}/month')
def monthly_visits(patient_id: str):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    visits = data[patient_id].get('visits', [])
    now = datetime.now()
    month_visits = [
        v for v in visits
        if datetime.fromisoformat(v).year == now.year and datetime.fromisoformat(v).month == now.month
    ]
    return {
        'patient_id': patient_id,
        'monthly_visits': len(month_visits)
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
    return JSONResponse(status_code=200, content={'message': 'patient updated with refreshed ML risk assessment'})

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
    """Clear all patient data - use with caution!"""
    save_data({})
    return {'message': 'All patient data cleared successfully'}

@app.get('/ml-model-info')
def get_ml_model_info():
    """Get information about the ML models"""
    return {
        'model_type': 'Random Forest Classifier',
        'models': {
            'risk_assessment': {
                'purpose': 'Predict patient risk level (High/Moderate/Low)',
                'features': ['age', 'height', 'weight', 'bmi', 'visit_count', 'gender', 'disease', 'blood_group'],
                'classes': ['High', 'Moderate', 'Low']
            },
            'treatment_prediction': {
                'purpose': 'Predict if patient needs treatment (Yes/No)',
                'features': ['age', 'height', 'weight', 'bmi', 'visit_count', 'gender', 'disease', 'blood_group'],
                'classes': ['Treatment Needed', 'No Treatment Needed']
            }
        },
        'training_data': '5000 synthetic patients with realistic health patterns',
        'accuracy_note': 'Models are trained on synthetic data for demonstration purposes',
        'risk_factors_analyzed': [
            'Age (>65 high risk, >50 moderate risk)',
            'BMI (>30 or <18.5 high risk, >25 moderate risk)',
            'Disease severity (chronic conditions = higher risk)',
            'Visit frequency (>10 visits = higher risk)'
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # Train ML models on startup
    print("Training ML models...")
    ml_model.train()
    print("ML models ready!")
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import re
import joblib
from pathlib import Path

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define request models
class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str = None

class PredictionResponse(BaseModel):
    predicted_role_logreg: str
    predicted_role_rf: str
    match_score: float
    missing_skills: list
    matched_jd_skills: list

# Initialize models and vectorizer
role_skills = {
    'Software Engineer': ['python', 'java', 'c++', 'algorithms', 'data structures', 
                          'oop', 'software development', 'debugging', 'git', 'docker'],
    'Data Scientist': ['python', 'machine learning', 'pandas', 'numpy', 'statistics',
                       'data analysis', 'scikit-learn', 'tensorflow', 'sql', 'matplotlib'],
    'Web Developer': ['javascript', 'html', 'css', 'react', 'node.js', 'express',
                      'mongodb', 'rest api', 'frontend', 'backend'],
    'UI/UX Designer': ['figma', 'adobe xd', 'user research', 'wireframing', 'prototyping',
                       'ui design', 'ux design', 'photoshop', 'illustrator', 'sketch'],
    'DevOps Engineer': ['aws', 'azure', 'docker', 'kubernetes', 'ci/cd', 'terraform',
                        'ansible', 'linux', 'bash', 'cloud computing'],
    'Mobile Developer': ['swift', 'kotlin', 'flutter', 'react native', 'mobile development',
                         'ios', 'android', 'xcode', 'android studio', 'mobile ui'],
    'Business Analyst': ['sql', 'excel', 'power bi', 'tableau', 'data analysis',
                         'requirements', 'stakeholder', 'uml', 'agile', 'project management'],
    'Database Administrator': ['sql', 'oracle', 'mysql', 'postgresql', 'database design',
                               'query optimization', 'indexing', 'backup', 'recovery', 'etl']
}

# Check if saved models exist
model_path = Path("models")
if not model_path.exists():
    model_path.mkdir()

vectorizer_path = model_path / "vectorizer.pkl"
logreg_path = model_path / "logreg.pkl"
rf_path = model_path / "randomforest.pkl"

if vectorizer_path.exists() and logreg_path.exists() and rf_path.exists():
    # Load existing models
    vectorizer = joblib.load(vectorizer_path)
    logreg = joblib.load(logreg_path)
    rf = joblib.load(rf_path)
else:
    # Generate synthetic data and train new models
    data = []
    for role, skills in role_skills.items():
        for _ in range(50):
            selected_skills = np.random.choice(skills, size=np.random.randint(5, 9), replace=False)
            text = ', '.join(selected_skills)
            data.append({'resume_text': text, 'job_role': role})

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
        df['resume_text'], df['job_role'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_vec, y_train)
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_vec, y_train)

    # Save models
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(logreg, logreg_path)
    joblib.dump(rf, rf_path)

# Helper functions
def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9.+-]*\b', text.lower())
    return set(words)

def filter_jd_skills(jd_text, role_skills):
    jd_kws = extract_keywords(jd_text)
    valid_skills = set()
    for skills in role_skills.values():
        valid_skills.update(skills)
    matched_skills = {kw for kw in jd_kws if kw in valid_skills}
    return matched_skills

# API Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_job_role(request: ResumeRequest):
    try:
        # Predict role
        vec = vectorizer.transform([request.resume_text])
        role_logreg = logreg.predict(vec)[0]
        role_rf = rf.predict(vec)[0]
        
        # Process job description if provided
        match_score = 0.0
        missing_skills = []
        matched_jd_skills = []
        
        if request.job_description:
            filtered_jd_skills = filter_jd_skills(request.job_description, role_skills)
            matched_jd_skills = list(filtered_jd_skills)
            
            if filtered_jd_skills:
                # Calculate match score
                resume_skills = extract_keywords(request.resume_text)
                match_score = len(resume_skills & filtered_jd_skills) / len(filtered_jd_skills) * 100
                
                # Find missing skills
                missing_skills = list(filtered_jd_skills - resume_skills)
        
        return {
            "predicted_role_logreg": role_logreg,
            "predicted_role_rf": role_rf,
            "match_score": round(match_score, 2),
            "missing_skills": missing_skills,
            "matched_jd_skills": matched_jd_skills
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/roles")
async def get_available_roles():
    return {"roles": list(role_skills.keys())}

@app.get("/skills/{role}")
async def get_skills_for_role(role: str):
    if role not in role_skills:
        raise HTTPException(status_code=404, detail="Role not found")
    return {"skills": role_skills[role]}




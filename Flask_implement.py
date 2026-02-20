
import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- 1. LOAD ASSETS (Using Absolute Paths for Windows) ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "Employee_Attrition_Risk_Analysis_&_Prediction.pkl")
ohe_path = os.path.join(base_path, "OneHotEncoder.pkl")

try:
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    with open(ohe_path, "rb") as f:
        OneHot_model = pickle.load(f)
    print("✅ Files loaded successfully from Deployment folder.")
except Exception as e:
    print(f"❌ Error loading pickle files: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        raw_input = request.get_json()
        user_df = pd.DataFrame([raw_input])

        # --- 2. ENCODING & TRANSFORMATION (Mirroring your Jupyter) ---
        user_df['Gender_encoded'] = user_df['Gender'].map({'Male': 1, 'Female': 0})
        user_df['OverTime_encoded'] = user_df['OverTime'].map({'Yes': 1, 'No': 0})
        user_df['BusinessTravel_encoded'] = user_df['BusinessTravel'].map({
             'Travel_Rarely': 1, 'Travel_Frequently': 0
        })

        cat_cols = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
        
        # This will work IF the file on disk is the one you fitted in Jupyter
        ohe_features = OneHot_model.transform(user_df[cat_cols])
        
        if hasattr(ohe_features, "toarray"):
            ohe_features = ohe_features.toarray()
            
        ohe_df = pd.DataFrame(ohe_features, columns=OneHot_model.get_feature_names_out(cat_cols))

        # Drop original text columns to match model training input
        user_numeric = user_df.drop(columns=cat_cols + ['Gender', 'OverTime', 'BusinessTravel'])
        user_final = pd.concat([user_numeric, ohe_df], axis=1)

        # --- 3. ALIGN COLUMNS & PREDICT ---
        # Crucial: Reindex to match the EXACT order and number of features in your XGBoost model
        user_final = user_final.reindex(columns=loaded_model.feature_names_in_, fill_value=0)

        prediction = loaded_model.predict(user_final)[0]
        probability = loaded_model.predict_proba(user_final)[0][1]

        return jsonify({
            "prediction": "High Risk" if int(prediction) == 1 else "Low Risk",
            "risk_percentage": f"{round(float(probability) * 100, 2)}%",
            "status": "Success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "Failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
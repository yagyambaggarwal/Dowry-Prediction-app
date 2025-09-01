from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load("RandomForestModel_Dowry_project.joblib")      
transformer = joblib.load("Encoder_Dowry_project.joblib") 

@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/calculator")
def calculator():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
 
    form_data = {
    "State": request.form["State"],
    "Caste": request.form["Caste"],
    "Religion": request.form["Religion"],
    "Groom_Age": int(request.form["Groom_Age"]),
    "Bride_Age": int(request.form["Bride_Age"]),
    "Groom_Education_Yrs": int(request.form["Groom_Education_Yrs"]),
    "Bride_Education_Yrs": int(request.form["Bride_Education_Yrs"]),
    "Groom_Income_Monthly": float(request.form["Groom_Income_Monthly"]),
    "Bride_Income_Monthly": float(request.form["Bride_Income_Monthly"]),
    "Occupation": request.form["Occupation"],
    "Marriage_Type": request.form["Marriage_Type"],
    "Area": request.form["Area"],
    "Family_Type": request.form["Family_Type"],
}


    df = pd.DataFrame([form_data])
    X_transformed = transformer.transform(df)
    prediction = model.predict(X_transformed)[0]

    return render_template("result.html", predicted=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)

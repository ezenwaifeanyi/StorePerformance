import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# Loap the pickle model 
model = pickle.load(open("spc_model.pk1", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    

    return render_template("index.html", prediction_text=" Store Performance Prediction {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

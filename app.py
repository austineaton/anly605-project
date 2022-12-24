import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(
    open(
        "StackedPickle.pkl",
        "rb"
    )
)


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    #TODO - convert categories back to numeric values using request.form.values()
    #use dictionary in stepII.ipynb to see categories corresponding numbers
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction[0] == 0:
        prediction = "<= 100K"
    else:
        prediction = "> 100K"
    return render_template(
        "index.html", prediction_text="Salary is {}".format(prediction)
    )


if __name__ == "__main__":
    flask_app.run(debug=True)
 
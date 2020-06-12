import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text = 'Employee Salary Should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug = True)
    
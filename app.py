from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
forest = pickle.load(open('heart.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict' , methods = ['GET', 'POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = forest.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('index.html', prediction_text= 'No_RISK')
    elif output == 1:
        return render_template('index.html', prediction_text='Stage 1')
    elif output == 2:
        return render_template('index.html', prediction_text='Stage 2')
    elif output == 3:
        return render_template('index.html', prediction_text='Stage 3')
    else:
        return render_template('index.html', prediction_text= 'Final Stage')


if __name__ == "__main__":
    app.run(debug=True)
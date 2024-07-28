from flask import Flask, request, render_template, abort 
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
loaded_model = joblib.load('fish_species_classifier.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = request.form.get('Weight')
    length1 = request.form.get('Length1')
    length2 = request.form.get('Length2')
    length3 = request.form.get('Length3')
    height = request.form.get('Height')
    width = request.form.get('width')

        # Check if any value is missing
    if not all([weight, length1, length2, length3, height, width]):
        abort(400, description="Missing values in the input form. Go back and re-enter all values for an accurate prediction.")
    
    form_data = pd.DataFrame({
    'Weight': [float(weight)],
    'Length1': [float(length1)],
    'Length2': [float(length2)],
    'Length3': [float(length3)],
    'Height': [float(height)],
    'Width': [float(width)],
    })
    
    prediction = loaded_model.predict(form_data)
    return render_template('inference.html', pred=prediction[0], form=form_data)
   


if __name__ == "__main__":
    app.run(debug=True)

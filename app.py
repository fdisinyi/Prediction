from flask import Flask, request, jsonify, render_template
import pickle
import json
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_age_salary = [int(x) for x in request.form.values()]
    user_age_salary1=[user_age_salary]
    scaled_result = scaler.transform(user_age_salary1)
    prediction = model.predict(scaled_result)
    if prediction==1:
        return render_template('index.html', prediction_text='Yes He will buy the Car')
    else:
        return render_template('index.html', prediction_text='No He do not buy the Car')
    

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    

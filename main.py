import pickle
import pandas as pd
from model_files.model import predict_data

from flask import Flask, request, jsonify

##creating a flask app and naming it "app"
app = Flask('app')

@app.route('/test')
def test():
    return 'Pinging Model Application!!'

@app.route('/predict', methods=['POST'])
def predict():
    data= request.get_json()
    df = pd.DataFrame.from_dict(data, orient='columns')
    print(df)

    path = r'C:\Users\antra\OneDrive - Queen Mary, University of London\Github\klarna_tech_assesment\model_files\lg_model3_pkl'
    
    with open(path, 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_data(df, model)

    result = {
        'test_prediction': list(predictions)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 9696)
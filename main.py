import os
import pickle
import pandas as pd
from model_files.model import predict_data
#from werkzeug import secure_filename
from flask import Flask, request, jsonify, render_template, redirect, url_for

path = os.getcwd()
os.chdir(path)
file_path=''

def predict(df):   
    
    #data= request.get_json()
    #df = pd.DataFrame.from_dict(data, orient='columns')
    uuid_list = df['uuid']
    #print(df)
    model_path = r'C:\Users\antra\OneDrive - Queen Mary, University of London\Github\klarna_tech_assesment\model_files\lg_model3_pkl'    
    with open(model_path, 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_data(df, model)

    result = {
        'uuid': list(uuid_list),
        'pd' : list(predictions)
    }
    return result

##creating a flask app and naming it "app"
app = Flask('app')

@app.route('/test')
def test():
    return ('Pinging Model Application!! from {}'.format(path))

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def get_data():
    if request.method == 'POST':
        file = request.files['file']   
        df = pd.read_csv(file)
        result= predict(df)
        res = pd.DataFrame.from_dict(result, orient='columns')
    
    return render_template('results.html', tables =res.to_dict(orient='records'))    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 9696)
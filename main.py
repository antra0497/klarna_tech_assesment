import os
import pickle
import pandas as pd
from model_files.model import predict_data
from flask import Flask, request, render_template, send_file, make_response

path = os.getcwd()
os.chdir(path)

def predict(df):   
    '''
    Input: Dataframe
    Output: Result dataframe, log file 
    '''
    uuid_list = df['uuid']  

    #defining the model file path  
    model_path = path + '\model_files\lg_model3_pkl' 

    #reading the model file
    with open(model_path, 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    #calling predict_data from model.py file       
    predictions, log = predict_data(df, model)

    #preparing result as json
    result = {
        'uuid': list(uuid_list),
        'pd' : list(predictions)
    }
    return result, log

##creating a flask app and naming it "app"
app = Flask('app')

@app.route('/test')
def test():
    return ('Pinging Model Application!! from {}'.format(path))

@app.route('/')
def form():
    #rendering the form.html as first page to take the input
    return render_template('form.html')

@app.route('/', methods=['POST', 'GET'])
def get_data():

    if request.method == 'POST':
        file = request.files['file']   
        #changing the file response into a dataframe
        df = pd.read_csv(file)
        #calling the predict function defined above
        result, log= predict(df)
        res = pd.DataFrame.from_dict(result, orient='columns')
        res.to_csv(path+'/static/output.csv') 
        #redenting the result.html with the result and the log 
        return render_template('results.html', tables =res.to_dict(orient='records'), logs =log)
 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 9696)
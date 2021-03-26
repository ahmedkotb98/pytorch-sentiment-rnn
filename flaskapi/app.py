import os
from flask import Flask,request,json
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
# LOAD MODEL
Model_path = 'model/caloriemodel11.pkl'

with open(Model_path,'rb') as filee:
  Model_ = pickle.load(filee)
  
ld = LabelEncoder()  

@app.route('/caloriemodel', methods=['GET', 'POST'])

def predict():
  try:
    inputs = request.args.get('inputs')
    details = np.mean(np.array(ld.fit_transform(inputs['details'])))
    quantity = np.mean(np.array(inputs['quantity']))
    reshape = [[details,quantity]]
    y_pred = Model_.predict(reshape)
    result_dict = {"result": y_pred}
    return app.response_class(
        response = json.dumps(result_dict, ensure_ascii=False),
        status = 200,
        mimetype='application/json'
    )
    except Exception:
      return app.response_class(
          response="Not Found",
          status=400,
      )


if __name__ == '__main__':
    app.run(debug=False)

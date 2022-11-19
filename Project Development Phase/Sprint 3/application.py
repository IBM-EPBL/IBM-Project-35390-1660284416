from flask import Flask,render_template,request
import pandas as pd
import pickle
app=Flask(__name__)
model=pickle.load(open("CarLinearRegressionModel.pkl",'rb'))
car=pd.read_csv("Cleaned Car.csv")

@app.route('/')
def index():
    
    car_models=sorted(car['car_models'].unique())
    company_name=sorted(car['company_name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    km_driven=sorted(car['year'].unique())
    fuel=sorted(car['fuel'].unique())
    seller_type=sorted(car['seller_type'].unique())
    transmission=sorted(car['transmission'].unique())
    owner=sorted(car['owner'].unique())
    return render_template('index.html',car_models=car_models,company_name=company_name,year=year,km_driven=km_driven,fuel=fuel,seller_type=seller_type,transmission=transmission,owner=owner)

@app.route('/predict',methods=['POST'])
def predict():
    company_name=request.form.get('company_name')
    car_models=request.form.get('car_models')
    year=int(request.form.get('year'))
    km_driven=int(request.form.get('kilo_driven'))
    fuel=request.form.get('fuel_type')
    seller_type=request.form.get('seller_type')
    transmission=request.form.get('transmission')
    owner=request.form.get('owner')
    prediction=model.predict(pd.DataFrame([[car_models,company_name,year,km_driven,fuel,seller_type,transmission,owner]],columns=['car_models','company_name','year','km_driven','fuel','seller_type','transmission','owner']))
    
    return str(prediction[0])
if __name__=="__main__":
    app.run(debug=True)
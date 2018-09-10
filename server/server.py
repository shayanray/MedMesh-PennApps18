import bson
import datetime
import fitbit
from flask import Flask, jsonify, render_template, request, Response
import logging
import os
import requests

# App
from configure import app

# add the mlpred folder
import sys
sys.path.insert(0, '../mlPredictor')
import predictionEngine
model = predictionEngine.train_model()

# Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('logs/server.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Modules
from modules import fitbit_module
from modules import betterdoctor

# MongoDB
from pymongo import MongoClient
client = MongoClient()
db = client.heartcare

# User Data
user_data = {
    'Amy': {
        'age': 38.0,
        'hypertension': 1.0,
        'heart_disease': 1.0,
        'bmi': 38.0,
        'gender_numeric': 1.0,
        'ever_married_numeric': 1.0,
        'work_type_numeric': 1.0,
        'residence_type_numeric': 1.0,
        'smoking_status_numeric': 1.0
    },
    'Bob': {
        'age': 78.0,
        'hypertension': 0.0,
        'heart_disease': 1.0,
        'bmi': 41.0,
        'gender_numeric': 1.0,
        'ever_married_numeric': 1.0,
        'work_type_numeric': 1.0,
        'residence_type_numeric': 1.0,
        'smoking_status_numeric': 1.0
    },
    'Charlie': {
        'age': 67.0,
        'hypertension': 0.0,
        'heart_disease': 1.0,
        'bmi': 38.0,
        'gender_numeric': 1.0,
        'ever_married_numeric': 1.0,
        'work_type_numeric': 1.0,
        'residence_type_numeric': 1.0,
        'smoking_status_numeric': 1.0
    }
}

'''
req >>  {'_id': ObjectId('5b94e647c5aef53ea07e43a5'),
'username': 'u33', 'password': 'sdsd', 'insurance': 'uhc',
'age': '12', 'bmi': 304687.5, 'height': 0.016, 'weight': '78',
'gender_numeric': '1.0', 'ever_married_numeric': '1.0', 'hypertension': '1.0',
'heart_disease': '1.0', 'smoking_status_numeric': '1.0', 'work_type_numeric': '2.0',
'residence_type_numeric': '2.0', 'state': 'ny', 'city': 'stonybrrok',
'datetime': datetime.datetime(2018, 9, 9, 5, 22, 15, 200000), 'heart_rate': 80.0}
'''


@app.route('/')
def register():
    return render_template('register.html')


@app.route('/api/signup', methods=['POST'])
def signup():
    args = request.form
    username = args['username']
    pw = args['password']
    insurance = args['insurance']
    age = args['age']
    height = float(args['height'])/100
    weight = args['weight']
    bmi = float(weight)/(float(height))**2
    gender = args['gender']
    married = args['married']
    hypertension = args['hypertension']
    heartdisease = args['heartdisease']
    smoking = args['smoking']
    worktype = args['worktype']
    residencetype = args['residencetype']
    state = args['state']
    city = args['city']
    dt = datetime.datetime.now()
    print("captured inputs.. about to insert into db")
    db.user.insert({'username': username, 'password': pw, 'insurance': insurance, 'age': age,
                    'bmi': bmi, 'height': height, 'weight': weight, 'gender_numeric': gender,
                    'ever_married_numeric': married, 'hypertension': hypertension,
                    'heart_disease': heartdisease, 'smoking_status_numeric': smoking,
                    'work_type_numeric': worktype, 'residence_type_numeric': residencetype,
                    'state': state, 'city': city, 'datetime': dt})
    return render_template('homepage.html')


@app.route('/api/user', methods=['POST'])
def get_judges():
    user = list(db.user.find().sort('datetime', -1))[0]
    print(user)
    if not user:
        logger.error('get_user({}): username not existed'.format(username))
        return None
    start = '13:00'
    end = '13:01'
    try:
        heart_rates = fitbit_module.get_heartrate(start=start, end=end)
        avg_hr = calculate_hr(heart_rates['activities-heart-intraday'])
    except Exception as e:
        avg_hr = 80.0
    req = user
    req['heart_rate'] = avg_hr

    # request json sync
    del req['_id']
    del req['username']
    del req['password']
    del req['insurance']
    del req['height']
    del req['weight']
    del req['city']
    del req['state']
    del req['datetime']

    print("deletion .. req >> ", req)
    # clear it for every request
    if 'stroke_probability' in req:
        del req['stroke_probability']
    stroke_probability = predictionEngine.predict(
        model, req)
    req['stroke_probability'] = stroke_probability
    hospital = getVisitType(req)
    if hospital == 'Heart Healthy':
        hospital = 'relax and no medical help needed'
    message = 'OK, here they are.\nBased on your heart rate in past month, the stroke probability is {}.\nWe recommend to go {}.\nWhere would you like to go?\n(Primary care, Urgent care, Emergency room, Nothing)'.format(
        round(stroke_probability, 2),
        hospital
    )
    res = {
        "user_id": "2",
        "bot_id": "1",
        "module_id": "3",
        "message": message,
        "stroke_probability": stroke_probability
    }
    return jsonify(res)


@app.route('/api/user/<username>', methods=['POST'])
def get_percentage(username):
    user = user_data.get(username)  # 'Amy'
    if not user:
        logger.error('get_user({}): username not existed'.format(username))
        return None
    if os.environ.get('env') == 'demo':
        # TODO: Get data from real cases
        pass
    else:
        if username == 'Charlie':
            avg_hr = 140.00
        elif username == 'Bob':
            avg_hr = 120.00
        else:
            avg_hr = 90.00
    req = user
    req['heart_rate'] = avg_hr
    # clear it for every request
    if 'stroke_probability' in req:
        del req['stroke_probability']

    stroke_probability = predictionEngine.predict(
        model, req)
    req['stroke_probability'] = probability
    hospital = getVisitType(req)
    if hospital == 'Heart Healthy':
        hospital = 'relax and no medical help needed'
    message = 'OK, here they are.\nBased on your heart rate in past month, the stroke probability is {}.\nWe recommend to go {}.\nWhere would you like to go?\n(Primary care, Urgent care, Emergency room, Nothing)'.format(
        round(probability, 2),
        hospital
    )
    res = {
        "user_id": "2",
        "bot_id": "1",
        "module_id": "3",
        "message": message,
        "stroke_probability": stroke_probability
    }
    return jsonify(res)


@app.route('/api/insurance_list', methods=['GET'])
def get_insurance_list():
    insurances = betterdoctor.getInsurances()
    insurance_list = []
    for insurance in insurances['data']:
        for plan in insurance['plans']:
            insurance_list.append({
                'name': plan['name'],
                'uid': plan['uid']
            })
    return jsonify(insurance_list)


@app.route('/api/specialty_list', methods=['GET'])
def get_specialty_list():
    specialties = betterdoctor.getSpecialties()
    specialty_list = []
    for specialty in specialties['data']:
        specialty_list.append({
            'uid': specialty['uid']
        })
    return jsonify(specialty_list)


@app.route('/api/insurance', methods=['GET', 'POST'])
def get_insurance():
    user = list(db.user.find().sort('datetime', -1))[0]
    location = '{}-{}'.format(
        user.get('state').lower(),
        user.get('city').lower()
    )
    insurance_name = user.get('insurance').lower()
    insurances = betterdoctor.getInsurances()
    insurance_list = []
    for insurance in insurances['data']:
        for plan in insurance['plans']:
            insurance_list.append({
                'name': plan['name'],
                'uid': plan['uid']
            })
    for insurance in insurance_list:
        if insurance_name in insurance['uid']:
            uid = insurance['uid']
    if not uid:
        uid = 'aetna-aetnabasichmo'
    doctors = betterdoctor.getDoctors(
        location=location, insurance=uid, specialty='cardiologist', limit=3
    )
    doctor_list = []
    for doctor in doctors['data']:
        doctor_list.append({
            'phone': doctor['practices'][0]['phones'][0]['number'],
            'street': doctor['practices'][0]['visit_address']['street'],
            'city': doctor['practices'][0]['visit_address']['city'],
            'name': doctor['practices'][0]['name'],
            'bio': doctor['profile']['bio'],
            'specialty': 'cardiologist'
        })
    map_base = 'https://www.google.com/maps/place/'
    google_map = map_base + \
        doctor_list[0]['street'].replace(
            ' ', '+') + '+' + doctor_list[0]['city']
    message = 'Here is the nearest doctor who covers your insurance network.\nBio: {} \nPhone Number: {} \nAddress: {} \nGoogle Map: {}'.format(
        doctor_list[0]['bio'],
        doctor_list[0]['phone'],
        doctor_list[0]['street'] + ', ' + doctor_list[0]['name'],
        google_map
    )
    res = {
        "user_id": "2",
        "bot_id": "1",
        "module_id": "3",
        "message": message
    }
    return jsonify(res)


def calculate_hr(heart_rates):
    total = 0
    for heart_rate in heart_rates['dataset']:
        total += heart_rate['value']
    avg = total/len(heart_rates['dataset'])
    return round(avg, 2)


def getVisitType(req):
    if req['stroke_probability'] < 0.20:
        return "Heart Healthy"
    elif req['stroke_probability'] >= 0.20 and req['stroke_probability'] < 0.4:
        return "Primary Care"
    elif req['stroke_probability'] >= 0.4 and req['stroke_probability'] < 0.7:
        return "Urgent Care"
    else:
        return "Emergency Room"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int("8080"), debug=True)

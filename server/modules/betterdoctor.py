import json
import os
import requests

BASE_URL = 'https://api.betterdoctor.com/2016-03-01'
key = os.environ.get('BETTERDOCTOR')


def _makeDict(**kwargs):
    requestDict = {}
    for key in kwargs:
        if kwargs[key] is not None:
            requestDict[key] = kwargs[key]
    return requestDict


def _returnOrExcept(res):
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        print(res.text)
        raise ValueError(str(res.status_code))


def getDoctors(name=None, firstName=None, lastName=None, query=None,
               specialty=None, insurance=None, practice=None, location=None,
               userLocation=None, gender=None, sort=None, fields=None, skip=0,
               limit=10):
    sort = 'distance-asc'
    requestDict = _makeDict(user_key=key, name=name, first_name=firstName,
                            last_name=lastName, query=query,
                            specialty_uid=specialty, insurance_uid=insurance,
                            practice_uid=practice, location=location,
                            user_location=userLocation, gender=gender,
                            sort=sort, fields=fields, skip=skip, limit=limit)
    response = requests.get(BASE_URL + "/doctors", params=requestDict)
    return _returnOrExcept(response)


def getPractices(limit=None, skip=None, sort=None, location=None, name=None):
    requestDict = _makeDict(user_key=key, limit=limit, skip=skip, sort=sort,
                            location=location, name=name)
    response = requests.get(BASE_URL + "/practices/", params=requestDict)
    return _returnOrExcept(response)


def getSpecialties(limit=None, skip=None, fields=None):
    requestDict = _makeDict(user_key=key, limit=limit, skip=skip,
                            fields=fields)
    response = requests.get(BASE_URL + "/specialties", params=requestDict)
    return _returnOrExcept(response)


def getInsurances(limit=None, skip=None, fields=None):
    requestDict = _makeDict(user_key=key, limit=limit, skip=skip,
                            fields=fields)
    response = requests.get(BASE_URL + "/insurances", params=requestDict)
    return _returnOrExcept(response)

import requests
import json

base_url = "http://20.51.231.48"

symptom_ids = [23158, 85193, 91606, 23157, 22977, 84409, 84060, 23833]

def call_symptom(symptom_ids):
    # Making a GET request to the /diagnosis output
    response = requests.get(f"{base_url}/symptom", params={"symptom_ids": symptom_ids})

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"
    
symptom_response = call_symptom(symptom_ids)
print("Symptom Response:", symptom_response)
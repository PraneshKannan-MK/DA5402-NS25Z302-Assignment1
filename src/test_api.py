import requests

URL = "http://127.0.0.1:5050/predict"

# Test 1 — Valid request
payload = {
    "features": [298, 308, 1500, 40, 10, 0, 0, 0, 0, 1, 0, 0]
}

r = requests.post(URL, json=payload)
print("Valid Request Status:", r.status_code)
print("Response:", r.text)

# Test 2 — Invalid length
payload = {"features": [1, 2]}

r = requests.post(URL, json=payload)
print("Invalid Length Status:", r.status_code)
print("Response:", r.text)

# Test 3 — Response format
payload = {
    "features": [298, 308, 1500, 40, 10, 0, 0, 0, 0, 1, 0, 0]
}

r = requests.post(URL, json=payload)

try:
    print("Keys:", r.json().keys())
except:
    print("Response not JSON")
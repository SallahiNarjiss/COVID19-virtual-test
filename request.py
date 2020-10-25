import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Fever':1, 'Tiredness':0,'Dry-Cough':1,'Difficulty-in-Breathing':0,'Sore-Throat':1,'None_Sympton':1,'Pains':1,'Nasal-Congestion':0,'Runny-Nose':1,'Diarrhea':1})

print(r.json())

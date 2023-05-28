import xgboost as xgb
import pandas as pd
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import load

num_to_key_dict = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
key_to_num_dict = {'B': 0, 'E': 1, 'F#': 2, 'A#': 3, 'A': 4, 'F': 5, 'C#': 6, 'D#': 7, 'D': 8, 'G': 9, 'C': 10, 'G#': 11}

class Model:
    def __init__(self, type: str):
        self.type = type
        if type == 'basic':
            self.model = xgb.XGBClassifier()
            self.model.load_model('basic_model.json')
            self.scaler = load('basic.bin')
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model('advanced_model.json')
            self.scaler = load('advanced.bin')


    def predict(self, X_test):
        if self.type == 'advanced':
            X_test["key"].replace(num_to_key_dict, inplace=True)
            X_test["key"].replace(key_to_num_dict, inplace=True)
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

model_basic = Model('basic')
model_advanced = Model('advanced')

df = pd.read_json("IUM23L_Zad_08_03_v2/tracks.jsonl", lines=True).drop(columns=["id", "name", "popularity", "explicit", "duration_ms", "id_artist", "release_date", "mode"])
rows_to_check = pd.DataFrame(df.iloc[1:5])
print('Rows to check: ', rows_to_check)

pred = model_basic.predict(rows_to_check)
print('Basic prediction: ', pred)

pred = model_advanced.predict(rows_to_check)
print('Advanced prediction: ', pred)
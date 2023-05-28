import sys

import joblib
import xgboost as xgb

sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import load

num_to_key_dict = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#",
                   11: "B"}
key_to_num_dict = {'B': 0, 'E': 1, 'F#': 2, 'A#': 3, 'A': 4, 'F': 5, 'C#': 6, 'D#': 7, 'D': 8, 'G': 9, 'C': 10,
                   'G#': 11}


class Model:
    def __init__(self, type: str):
        self.type = type
        if type == 'basic':
            self.model = xgb.XGBClassifier()
            self.model.load_model('model_setup/basic_model.json')
            self.scaler = load('model_setup/basic.bin')
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model('model_setup/advanced_model.json')
            self.scaler = load('model_setup/advanced.bin')

    def predict(self, X_test):
        X_test = X_test.drop(
            columns=["id", "name", "popularity", "explicit", "duration_ms", "id_artist", "release_date"])
        if self.type == 'advanced':
            X_test["key"].replace(num_to_key_dict, inplace=True)
            X_test["key"].replace(key_to_num_dict, inplace=True)
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

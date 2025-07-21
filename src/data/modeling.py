import pandas as pd
import numpy as np 
import pickle 

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/interim/train_bow.csv")

x_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

pickle.dump(model, open("models/random_forest_model.pkl", "wb"))

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Name"
]

Y_COL = "Transported"


def main():
    df = pd.read_csv("./.data/test.csv", index_col=False)
    df = df.fillna("-999")

    model = pickle.load(open("./.model/model.pkl", "rb"))
    encoder = LabelEncoder()
    df["HomePlanet"] = encoder.fit_transform(df['HomePlanet'])
    df["Cabin"] = encoder.fit_transform(df["Cabin"])
    df["Destination"] = encoder.fit_transform(df["Destination"])
    df["Name"] = encoder.fit_transform(df["Name"])
    df["Transported"] = model.predict_proba(df[FEATURES])[:, -1]
    df.to_csv("./.data/scored.csv")

if __name__ == "__main__":
    main()
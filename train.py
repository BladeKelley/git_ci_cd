import pandas as pd
import pickle
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
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
    df = pd.read_csv("./.data/train.csv", index_col=False)
    df = df.fillna("-999")
    encoder = LabelEncoder()
    df["HomePlanet"] = encoder.fit_transform(df['HomePlanet'])
    df["Cabin"] = encoder.fit_transform(df["Cabin"])
    df["Destination"] = encoder.fit_transform(df["Destination"])
    df["Name"] = encoder.fit_transform(df["Name"])

    model = RandomForestClassifier(
        n_estimators=50, min_samples_leaf=400, max_depth=8, verbose=1, n_jobs=-1
    )
    model.fit(
        df[FEATURES], df[Y_COL]
    )
    feature_importances = pd.DataFrame(
        model.feature_importances_, index=FEATURES, columns=["importance"]
    ).sort_values("importance", ascending=False)
    feature_importances.to_csv(
        "./.model/feature_importance.csv"
    )
    pickle.dump(model, open('.model/model.pkl', 'wb'))


if __name__ == "__main__":
    main()
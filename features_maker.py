# feature_maker.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0

import pandas as pd

def load_city_reference():
    return pd.DataFrame({
        "city": [
            "Paris", "Marseille", "Lyon", "Toulouse", "Nice",
            "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille",
            "Rennes", "Reims", "Le Havre", "Saint-Étienne", "Toulon",
            "Grenoble", "Dijon", "Angers", "Nîmes", "Villeurbanne",
            "Clermont-Ferrand", "Aix-en-Provence", "Brest", "Limoges",
            "Tours", "Amiens", "Metz", "Perpignan", "Besançon",
            "Orléans", "Mulhouse"
        ],
        "lat": [
            48.8566, 43.2964, 45.7634, 43.6045, 43.7031,
            47.2181, 48.5800, 43.6119, 44.8361, 50.6292,
            48.1147, 49.2628, 49.4900, 45.4347, 43.1258,
            45.1715, 47.3167, 47.4736, 43.8380, 45.7667,
            45.7831, 43.5263, 48.3899, 45.8353, 47.2436,
            49.8941, 49.1203, 42.6986, 47.2400, 47.9025,
            47.7508
        ],
        "lon": [
            2.3522, 5.3700, 4.8343, 1.4440, 7.2661,
            -1.5528, 7.7500, 3.8772, -0.5808, 3.0573,
            -1.6794, 4.0347, 0.1000, 4.3903, 5.9306,
            5.7224, 5.0167, -0.5542, 4.3610, 4.8803,
            3.0824, 5.4454, -4.4900, 1.2625, 0.6892,
            2.2958, 6.1778, 2.8956, 6.0200, 1.9090,
            7.3359
        ],
        "population": [
            2120000, 875000, 523000, 512000, 348000,
            323000, 286000, 298000, 262000, 235000,
            225000, 182000, 168000, 173000, 174000,
            158000, 155000, 158000, 148000, 163000,
            145000, 148000, 147000, 135000, 138000,
            140000, 123000, 112000, 125000, 120000,
            112000
        ],
        "rank": list(range(1, 32))
    })



class NearestCityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10):
        self.top_n = top_n

    def fit(self, X, y=None):
        cities = load_city_reference().sort_values("rank").head(self.top_n)
        coords = np.deg2rad(cities[["lat", "lon"]].values)
        self.tree_ = BallTree(coords, metric="haversine")

        self.city_rank_ = cities["rank"].values
        self.city_pop_ = cities["population"].values

        return self

    def transform(self, X):
        coords = np.deg2rad(X[["lat", "lon"]].values)
        dist, ind = self.tree_.query(coords, k=1)

        return np.column_stack([
            dist[:, 0] * 6371.0,
            self.city_rank_[ind[:, 0]],
            np.log1p(self.city_pop_[ind[:, 0]])
        ])


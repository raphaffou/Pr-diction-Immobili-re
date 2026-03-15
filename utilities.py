import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pertinent_features = [
    "city", "postal_code", "department_code", "lat", "lon", "surface",
    "rooms", "bedrooms", "bathrooms", "showers", "floor", "balconies",
    "terraces", "parking_places", "cellars", "elevator", "disabled_access",
    "furnished", "duplex", "energy_class", "energy_value", "ghg_class",
    "ghg_value", "energy_min_cost", "energy_max_cost", "heating_type",
    "new_property", "description", "price"
]

""""
Fonction de formating des données du CSV
@params : csv : fichier de csv extrait grâce à l'API de Bien'ici.com
"""
def formatData(csv):
    df = pd.read_csv(csv)
    df = df[pertinent_features].copy()
    df.loc[df["duplex"].isna(), "duplex"] = False
    df = df.dropna().copy()
    return df
    

"""
Fonction de génération d'une carte des loyers en France en 3D
(coordonnées restreintes à la France)
@params : df1 : correspond à un dataframe (contenant au moins les champs suivants :
                    - lon (longitude)
                    - lat (latitude)
                    - price (loyer)
"""
def map3DFrance(df1):
    df_filtered = df1[
        (df1["lat"] >= 40) & (df1["lat"] <= 60) &
        (df1["lon"] >= -20) & (df1["lon"] <= 20)
    ]
    
    x = df_filtered["lon"]
    y = df_filtered["lat"]
    z = df_filtered["price"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x, y, z, s=20)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Price")
    
    plt.show()

"""
Fonction de génération d'une carte des loyers en France en 2D
(coordonnées restreintes à la France)
@params : df1 : correspond à un dataframe (contenant au moins les champs suivants :
                    - lon (longitude)
                    - lat (latitude)
                    - price (loyer)
"""
def map2DFrance(df1):
    df_filtered = df1[
    (df1["lat"] >= 40) & (df1["lat"] <= 60) &
    (df1["lon"] >= -20) & (df1["lon"] <= 20)
    ]
    
    x = df_filtered["lon"]
    y = df_filtered["lat"]
    price = df_filtered["price"]/df_filtered["surface"]
    
    url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
    france = gpd.read_file(url)
    fig, ax = plt.subplots(figsize=(12, 12)) # J'ai agrandi un peu la figure
    
    france.plot(ax=ax, color='white', edgecolor='grey', alpha=0.5, zorder=1)
    
    sc = ax.scatter(
        x,
        y,
        c=price,
        cmap="viridis",
        s=20,
        zorder=2,
        alpha=0.8
    )
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(sc, label="Price")
    
    ax.set_xlim(-5, 10)
    ax.set_ylim(41, 52)
    
    plt.title("Prix en France (Détail Départements)")
    plt.show()

"""
Fonction de génération d'une carte des loyers à Paris en 2D
(coordonnées restreintes à la ville de Paris)
@params : df1 : correspond à un dataframe (contenant au moins les champs suivants :
                    - lon (longitude)
                    - lat (latitude)
                    - price (loyer)
"""
def map2DParis(df1):
    df_filtered = df1[
    (df1["lat"] >= 48.5) & (df1["lat"] <= 49) &
    (df1["lon"] >= 1.2) & (df1["lon"] <= 4)
    ]
    
    geometry = [Point(lon, lat) for lon, lat in zip(df_filtered["lon"], df_filtered["lat"])]
    gdf = gpd.GeoDataFrame(df_filtered, geometry=geometry, crs="EPSG:4326")  # WGS84 Lat/Lon
    
    # Convert to Web Mercator for contextily
    gdf = gdf.to_crs(epsg=3857)
    
    # Compute price per m² for coloring
    gdf["price_per_m2"] = gdf["price"] / gdf["surface"]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot of the points, colored by price per m²
    gdf.plot(
        ax=ax,
        column="price_per_m2",
        cmap="viridis",
        markersize=20,
        legend=True,
        legend_kwds={'label': "Price per m² (€)", 'shrink': 0.6}
    )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Loyer par mètre carré à Paris")
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()

"""
Fonction d'affichage des données prédites
@params :   y_test_pca : les données testées
            y_pred_pca : les données entrainées
"""
def predictData(y_test_pca,y_pred_pca):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_pca, y=y_pred_pca)
    plt.plot([y_test_pca.min(), y_test_pca.max()], [y_test_pca.min(),
                y_test_pca.max()], 'k--', lw=2, color='red')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices (PCA Model)")
    plt.show()

def create_dataframe_from_input():
    data = {}
    for field in pertinent_features:

        if field == "price":
            continue
        
        value = input(f"Enter {field}: ")

        
        # Convert numeric fields automatically
        if field in ["lat", "lon", "surface", "rooms", "bedrooms", "bathrooms", 
                     "showers", "floor", "balconies", "terraces", "parking_places", 
                     "cellars", "energy_value", "ghg_value", "energy_min_cost", 
                     "energy_max_cost", "price"]:
            try:
                value = float(value)
            except ValueError:
                print(f"Warning: {field} could not be converted to float. Storing as string.")
        
        # Convert boolean fields
        if field in ["elevator", "disabled_access", "furnished", "duplex", "new_property"]:
            if value.lower() in ["yes", "true", "1"]:
                value = True
            elif value.lower() in ["no", "false", "0"]:
                value = False
            else:
                value = None  # if input is not clear
        
        data[field] = [value]  # wrap in list for DataFrame
    
    df = pd.DataFrame(data)
    return df

"""
Utilities pour le projet de prédiction de loyer
Raphaël Thieffry, Antonin Russac, Ethan Puyaubreau
"""


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(csv_path)


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Retourne les colonnes de features et de target."""
    pertinent_features = [
        "city", "postal_code", "department_code", "lat", "lon", 
        "surface", "rooms", "bedrooms", "bathrooms", "showers", 
        "floor", "balconies", "terraces", "parking_places", "cellars", 
        "elevator", "disabled_access", "furnished", "duplex", 
        "energy_class", "energy_value", "ghg_class", "ghg_value", 
        "energy_min_cost", "energy_max_cost", "heating_type", 
        "new_property", "description"
    ]
    target_features = ["price"]
    return pertinent_features, target_features


# =============================================================================
# PRÉTRAITEMENT DES DONNÉES
# =============================================================================

def preprocess_data(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Prétraite les données:
    - Sélectionne les colonnes pertinentes
    - Remplit les valeurs manquantes pour duplex
    - Supprime les lignes avec des NaN
    """
    df_processed = df[features].copy()
    
    # Remplir les duplex manquants avec False
    if "duplex" in df_processed.columns:
        df_processed.loc[df_processed["duplex"].isna(), "duplex"] = False
    
    # Supprimer les lignes avec des NaN
    df_processed = df_processed.dropna().copy()
    
    return df_processed


def filter_geographic_data(df: pd.DataFrame, 
                           lat_range: tuple = (40, 60), 
                           lon_range: tuple = (-20, 20)) -> pd.DataFrame:
    """Filtre les données selon les coordonnées géographiques."""
    return df[
        (df["lat"] >= lat_range[0]) & (df["lat"] <= lat_range[1]) &
        (df["lon"] >= lon_range[0]) & (df["lon"] <= lon_range[1])
    ]


def encode_categorical_features(df: pd.DataFrame, 
                                 categorical_cols: list[str] = None) -> tuple[pd.DataFrame, dict]:
    """
    Encode les colonnes catégorielles avec LabelEncoder.
    Retourne le DataFrame encodé et un dictionnaire des encoders.
    """
    if categorical_cols is None:
        categorical_cols = ["city", "department_code", "energy_class", 
                           "ghg_class", "heating_type"]
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Convertir en string pour gérer les types mixtes
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def prepare_ml_features(df: pd.DataFrame, 
                        target_col: str = "price",
                        exclude_cols: list[str] = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les features pour le machine learning.
    Exclut les colonnes non numériques et les colonnes spécifiées.
    """
    if exclude_cols is None:
        exclude_cols = ["description", "city"]  # Description est du texte
    
    # Exclure les colonnes spécifiées et la target
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    # Convertir les booléens en int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    return X, y


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_3d_price_scatter(df: pd.DataFrame, 
                          lat_range: tuple = (40, 60), 
                          lon_range: tuple = (-20, 20)) -> None:
    """Affiche un scatter plot 3D des prix en fonction de lat/lon."""
    df_filtered = filter_geographic_data(df, lat_range, lon_range)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df_filtered["lon"], df_filtered["lat"], df_filtered["price"], s=20)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Price")
    ax.set_title("Prix des loyers en 3D")
    
    plt.show()


def plot_france_map(df: pd.DataFrame, 
                    lat_range: tuple = (40, 60), 
                    lon_range: tuple = (-20, 20),
                    price_per_m2: bool = True) -> None:
    """Affiche une carte de France avec les prix."""
    df_filtered = filter_geographic_data(df, lat_range, lon_range)
    
    x = df_filtered["lon"]
    y = df_filtered["lat"]
    
    if price_per_m2:
        price = df_filtered["price"] / df_filtered["surface"]
        label = "Prix par m²"
    else:
        price = df_filtered["price"]
        label = "Prix"
    
    # Charger le contour de la France
    url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
    france = gpd.read_file(url)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    france.plot(ax=ax, color='white', edgecolor='grey', alpha=0.5, zorder=1)
    
    sc = ax.scatter(x, y, c=price, cmap="viridis", s=20, zorder=2, alpha=0.8)
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(sc, label=label)
    
    ax.set_xlim(-5, 10)
    ax.set_ylim(41, 52)
    
    plt.title(f"{label} en France (Détail Départements)")
    plt.show()


def plot_city_map(df: pd.DataFrame, 
                  lat_range: tuple = (48.5, 49), 
                  lon_range: tuple = (1.2, 4),
                  title: str = "Loyer par mètre carré") -> None:
    """Affiche une carte détaillée d'une ville avec fond de carte."""
    df_filtered = filter_geographic_data(df, lat_range, lon_range)
    
    geometry = [Point(lon, lat) for lon, lat in zip(df_filtered["lon"], df_filtered["lat"])]
    gdf = gpd.GeoDataFrame(df_filtered, geometry=geometry, crs="EPSG:4326")
    
    # Convertir en Web Mercator pour contextily
    gdf = gdf.to_crs(epsg=3857)
    gdf["price_per_m2"] = gdf["price"] / gdf["surface"]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    gdf.plot(
        ax=ax,
        column="price_per_m2",
        cmap="viridis",
        markersize=20,
        legend=True,
        legend_kwds={'label': "Prix par m² (€)", 'shrink': 0.6}
    )
    
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()


def plot_feature_importance(model, feature_names: list[str], top_n: int = 15) -> None:
    """Affiche l'importance des features pour un modèle."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("Le modèle ne supporte pas l'affichage des importances.")
        return
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title("Importance des Features")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Affiche un graphique comparant les prédictions aux valeurs réelles."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("Prédictions vs Valeurs réelles")
    plt.tight_layout()
    plt.show()


# =============================================================================
# MODÈLES DE PRÉDICTION
# =============================================================================

def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series, 
                model_type: str = "random_forest",
                **kwargs) -> tuple:
    """
    Entraîne un modèle de régression.
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        model_type: Type de modèle ('random_forest', 'gradient_boosting', 'linear', 'ridge')
        **kwargs: Arguments supplémentaires pour le modèle
    
    Returns:
        Tuple (modèle entraîné, scaler)
    """
    # Normaliser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Sélectionner le modèle
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=kwargs.get('alpha', 1.0))
    else:
        raise ValueError(f"Model type '{model_type}' non reconnu.")
    
    model.fit(X_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Évalue les performances d'un modèle.
    
    Returns:
        Dictionnaire avec les métriques (MSE, RMSE, MAE, R²)
    """
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return metrics, y_pred


def predict_price(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Prédit le prix pour de nouvelles données."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def print_metrics(metrics: dict) -> None:
    """Affiche les métriques de manière formatée."""
    print("=" * 50)
    print("MÉTRIQUES DU MODÈLE")
    print("=" * 50)
    print(f"MSE  (Mean Squared Error):     {metrics['mse']:,.2f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:,.2f} €")
    print(f"MAE  (Mean Absolute Error):     {metrics['mae']:,.2f} €")
    print(f"R²   (Coefficient de détermination): {metrics['r2']:.4f}")
    print("=" * 50)


def compare_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Compare plusieurs modèles et retourne un DataFrame des résultats."""
    models = {
        'Linear Regression': 'linear',
        'Ridge Regression': 'ridge',
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting'
    }
    
    results = []
    
    for name, model_type in models.items():
        print(f"Entraînement de {name}...")
        model, scaler = train_model(X_train, y_train, model_type=model_type)
        metrics, _ = evaluate_model(model, scaler, X_test, y_test)
        
        results.append({
            'Modèle': name,
            'RMSE (€)': metrics['rmse'],
            'MAE (€)': metrics['mae'],
            'R²': metrics['r2']
        })
    
    return pd.DataFrame(results).sort_values('R²', ascending=False)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score, root_mean_squared_error

def cross_val_price_metrics_from_log(
    model,
    X,
    y,
    n_splits=5,
    random_state=42
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = clone(model)
        m.fit(X_train, np.log(y_train))

        y_pred_log = m.predict(X_val)

        
        y_pred = np.exp(y_pred_log)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse = root_mean_squared_error(y_val, y_pred)
        rmse_scores.append(rmse*rmse)

    return {
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "r2_folds": r2_scores,
        "rmse_folds": rmse_scores,
    }


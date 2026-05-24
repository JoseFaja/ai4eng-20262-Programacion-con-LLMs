import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detectar_fallos_maquinaria(df, tamano_ventana, tasa_contaminacion):
    if tamano_ventana > len(df):
        raise ValueError("La ventana es mayor que los datos disponibles")

    suavizado = df['temperatura'].rolling(window=tamano_ventana).mean().dropna()
    X_suavizado = suavizado.values.reshape(-1, 1)

    iso_forest = IsolationForest(
        contamination=tasa_contaminacion,
        random_state=42
    )

    predicciones = iso_forest.fit_predict(X_suavizado)

    conteo_anomalias = int(np.sum(predicciones == -1))
    tasa_real = float(conteo_anomalias / len(predicciones))

    return {
        "total_anomalias": conteo_anomalias,
        "tasa_real_anomalias": tasa_real
    }
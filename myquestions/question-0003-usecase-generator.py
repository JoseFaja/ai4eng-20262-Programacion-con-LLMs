import numpy as np
import pandas as pd
import random
from sklearn.linear_model import Ridge

def generar_caso_de_uso_weighted_regression_score():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función weighted_regression_score.
    """

    
    # 1. Configuración aleatoria
  
    n_rows = random.randint(30, 80)
    n_features = random.randint(2, 5)

    # 2. Generación de datos
    
    X = np.random.randn(n_rows, n_features)

    true_coef = np.random.randn(n_features)
    noise = np.random.normal(0, 0.5, size=n_rows)

    y = X @ true_coef + noise

    weights = np.random.uniform(0.5, 2.0, size=n_rows)

    
    # 3. INPUT
    
    input_data = {
        'X': X.copy(),
        'y': y.copy(),
        'weights': weights.copy()
    }

    
    # 4. OUTPUT ESPERADO
   
    model = Ridge()
    model.fit(X, y)

    y_pred = model.predict(X)

    wmse = np.sum(weights * (y - y_pred) ** 2) / np.sum(weights)

    output_data = float(wmse)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_weighted_regression_score()

    print("=== INPUT ===")
    print("Shape X:", entrada['X'].shape)
    print("Shape y:", entrada['y'].shape)
    print("Shape weights:", entrada['weights'].shape)

    print("\n=== OUTPUT ESPERADO ===")
    print("Weighted MSE:", salida_esperada)
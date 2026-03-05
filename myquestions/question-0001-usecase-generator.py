import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_detectar_top_percentile():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función detectar_top_percentile.
    """

    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    n_rows = random.randint(30, 80)
    categories = ['A', 'B', 'C', 'D']
    percentile = random.choice([70, 75, 80, 85, 90])

    # ---------------------------------------------------------
    # 2. Generación de datos
    # ---------------------------------------------------------
    df = pd.DataFrame({
        'category': np.random.choice(categories, size=n_rows),
        'value': np.random.normal(loc=50, scale=15, size=n_rows)
    })

    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'percentile': percentile
    }

    # ---------------------------------------------------------
    # 4. OUTPUT ESPERADO
    # ---------------------------------------------------------
    thresholds = df.groupby('category')['value'].quantile(percentile / 100)
    df_temp = df.join(thresholds, on='category', rsuffix='_threshold')
    count_top = (df_temp['value'] >= df_temp['value_threshold']).sum()

    output_data = int(count_top)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_detectar_top_percentile()

    print("=== INPUT ===")
    print("Percentile:", entrada['percentile'])
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    print("Total top rows:", salida_esperada)
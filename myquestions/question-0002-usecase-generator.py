import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_entropia_por_grupo():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_entropia_por_grupo.
    """

    
    # 1. Configuración aleatoria
    
    n_rows = random.randint(40, 100)
    groups = ['G1', 'G2', 'G3', 'G4']
    classes = ['X', 'Y', 'Z']

    
    # 2. Generación de datos
   
    df = pd.DataFrame({
        'group': np.random.choice(groups, size=n_rows),
        'class_label': np.random.choice(classes, size=n_rows)
    })

   
    # 3. INPUT
   
    input_data = {
        'df': df.copy()
    }

   
    # 4. OUTPUT ESPERADO
    
    entropy_results = []

    for g in sorted(df['group'].unique()):
        subset = df[df['group'] == g]
        probs = subset['class_label'].value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        entropy_results.append((g, entropy))

    expected_df = pd.DataFrame(entropy_results, columns=['group', 'entropy'])

    output_data = expected_df

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_calcular_entropia_por_grupo()

    print("=== INPUT ===")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
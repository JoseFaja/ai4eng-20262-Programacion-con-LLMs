import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_matriz_confusion_normalizada():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función matriz_confusion_normalizada.
    """

   
    # 1. Configuración aleatoria
    
    n_rows = random.randint(60, 120)
    n_features = random.randint(3, 6)
    n_classes = random.randint(3, 4)

    
    # 2. Generación de datos
   
    X = np.random.randn(n_rows, n_features)

    # Generamos etiquetas multiclase
    y = np.random.randint(0, n_classes, size=n_rows)

    # Aseguramos que todas las clases aparezcan al menos una vez
    for i in range(n_classes):
        y[i] = i

   
    # 3. INPUT
 
    input_data = {
        'X': X.copy(),
        'y': y.copy()
    }

    
    # 4. OUTPUT ESPERADO
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Forzamos dimensiones consistentes
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))

    # Normalización por filas (evitando división por cero)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm / row_sums

    output_data = cm_normalized

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_matriz_confusion_normalizada()

    print("=== INPUT ===")
    print("Shape X:", entrada['X'].shape)
    print("Shape y:", entrada['y'].shape)

    print("\n=== OUTPUT ESPERADO ===")
    print("Shape matriz:", salida_esperada.shape)
    print(salida_esperada)
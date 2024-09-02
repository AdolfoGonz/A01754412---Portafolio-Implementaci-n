import numpy as np
import pandas as pd
from model.model_train import build_tree, predict_tree


"""
Esta función carga un dataset desde un archivo CSV y lo prepara para entrenar un modelo de árbol de decisión.

El dataset se carga en un DataFrame de pandas, se separan las características de la etiqueta (quality), y luego se divide en un conjunto de entrenamiento (60%) para entrenar el modelo.

Finalmente, se construye y retorna un árbol de decisión usando los datos de entrenamiento, con una profundidad máxima especificada para evitar el sobreajuste.

"""


def load_trained_tree():

    data = pd.read_csv(
        '/Users/aguero/Desktop/A01754412---Portafolio-Implementaci-n/Data/normalized_winequality_dataset.csv')

    X = data.drop('quality', axis=1).values
    y = data['quality'].values

    train_size = int(0.6 * len(data))
    X_train, y_train = X[:train_size], y[:train_size]

    decision_tree = build_tree(X_train, y_train, max_depth=5)
    return decision_tree


"""
Este array contiene las características de un vino específico, que se han introducido manualmente.

Cada valor en el array representa una característica específica del vino, como acidez, azúcar residual,pH, entre otras.
"""
features = np.array([1.654856079, 0.794282374, 1.432803137, -0.169427234, -0.073676911, -
                    0.944346356, -1.017721, 1.724304591, -0.91431164, 0.305989631, -0.866378858])

"""
Definimos las medias y desviaciones estándar para cada característica del dataset de entrenamiento.

Estos valores se utilizan para normalizar las nuevas entradas (features) de modo que estén en la misma escala que los datos de entrenamiento. Esto es esencial para asegurar que el modelo de predicción funcione correctamente.
"""

mean = np.array([8.319637, 0.527821, 0.270976, 2.538806, 0.087467,
                15.874922, 46.467792, 0.996747, 3.311113, 0.658149, 10.422983])
std = np.array([1.741096, 0.179060, 0.194801, 1.409928, 0.047065,
               10.460157, 32.895324, 0.001887, 0.154386, 0.169507, 1.065668])

features = (features - mean) / std

"""

Se carga el modelo de árbol de decisión que ha sido previamente entrenado utilizando los datos de entrenamiento.
Luego, se utiliza este modelo para predecir la calidad del vino basándose en las características normalizadas.

"""
trained_tree = load_trained_tree()
prediction = predict_tree(trained_tree, features)

"""
Dependiendo del valor predicho, se imprime si el vino es bueno o malo.

En este caso, asumimos que una predicción de '1' indica que el vino es bueno, mientras que cualquier otro valor indica que es malo.
"""

if prediction == 1:
    print("El vino es bueno.")
else:
    print("El vino es malo.")

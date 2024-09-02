import numpy as np
import pandas as pd
from model.model_train import build_tree, predict_tree

# Cargar el árbol de decisión ya entrenado o entrenarlo de nuevo


def load_trained_tree():
    # Cargar el dataset desde el CSV
    data = pd.read_csv(
        '/Users/aguero/Desktop/A01754412---Portafolio-Implementaci-n/Data/normalized_winequality_dataset.csv')

    # Separar las características y la etiqueta
    X = data.drop('quality', axis=1).values
    y = data['quality'].values

    # Dividir el dataset en train (60%)
    train_size = int(0.6 * len(data))
    X_train, y_train = X[:train_size], y[:train_size]

    # Construir el árbol de decisión
    decision_tree = build_tree(X_train, y_train, max_depth=5)
    return decision_tree


# Ingresar las características del vino manualmente
features = np.array([1.654856079, 0.794282374, 1.432803137, -0.169427234, -0.073676911, -
                    0.944346356, -1.017721, 1.724304591, -0.91431164, 0.305989631, -0.866378858])


# Normalización (si es necesario)
mean = np.array([8.319637, 0.527821, 0.270976, 2.538806, 0.087467,
                15.874922, 46.467792, 0.996747, 3.311113, 0.658149, 10.422983])
std = np.array([1.741096, 0.179060, 0.194801, 1.409928, 0.047065,
               10.460157, 32.895324, 0.001887, 0.154386, 0.169507, 1.065668])
features = (features - mean) / std

# Cargar el árbol entrenado y hacer predicción
trained_tree = load_trained_tree()
prediction = predict_tree(trained_tree, features)

# Mostrar el resultado
if prediction == 1:
    print("El vino es bueno.")
else:
    print("El vino es malo.")

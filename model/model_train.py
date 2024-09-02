import numpy as np
import pandas as pd

# Función para calcular la impureza de Gini


def gini_impurity(y):
    classes = np.unique(y)
    gini = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        gini -= p ** 2
    return gini

# Función para calcular la ganancia de información


def information_gain(y, left_y, right_y):
    p = len(left_y) / len(y)
    return gini_impurity(y) - (p * gini_impurity(left_y) + (1 - p) * gini_impurity(right_y))

# Función para encontrar la mejor división de los datos


def best_split(X, y):
    best_gain = 0
    best_split = None
    for column in range(X.shape[1]):
        unique_values = np.unique(X[:, column])
        for val in unique_values:
            left_mask = X[:, column] <= val
            right_mask = X[:, column] > val
            left_y = y[left_mask]
            right_y = y[right_mask]
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_gain = gain
                best_split = {
                    'column': column,
                    'value': val,
                    'left_mask': left_mask,
                    'right_mask': right_mask
                }
    return best_gain, best_split

# Función para construir el árbol de decisión


def build_tree(X, y, depth=0, max_depth=5):
    if depth == max_depth or len(np.unique(y)) == 1:
        return np.argmax(np.bincount(y))

    gain, split = best_split(X, y)
    if gain == 0:
        return np.argmax(np.bincount(y))

    left_tree = build_tree(X[split['left_mask']],
                           y[split['left_mask']], depth + 1, max_depth)
    right_tree = build_tree(X[split['right_mask']],
                            y[split['right_mask']], depth + 1, max_depth)

    return {
        'column': split['column'],
        'value': split['value'],
        'left': left_tree,
        'right': right_tree
    }

# Función para hacer predicciones usando el árbol de decisión


def predict_tree(tree, X):
    if isinstance(tree, dict):
        if X[tree['column']] <= tree['value']:
            return predict_tree(tree['left'], X)
        else:
            return predict_tree(tree['right'], X)
    else:
        return tree

# Función para hacer predicciones en un conjunto de datos


def predict_tree_batch(tree, X):
    return np.array([predict_tree(tree, row) for row in X])

# Evaluación del modelo


def evaluate_performance(actual_labels, predicted_labels):
    accuracy = np.mean(actual_labels == predicted_labels)

    # Evitar división por cero
    if np.sum(predicted_labels == 1) == 0:
        precision = 0
    else:
        precision = np.sum((actual_labels == 1) & (
            predicted_labels == 1)) / np.sum(predicted_labels == 1)

    if np.sum(actual_labels == 1) == 0:
        recall = 0
    else:
        recall = np.sum((actual_labels == 1) & (
            predicted_labels == 1)) / np.sum(actual_labels == 1)

    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    conf_matrix = pd.crosstab(np.array(actual_labels), np.array(
        predicted_labels), rownames=['Actual'], colnames=['Predicted'])

    return accuracy, precision, recall, f1_score, conf_matrix


# Cargar el dataset desde el CSV
data = pd.read_csv(
    '/Users/aguero/Desktop/A01754412---Portafolio-Implementaci-n/Data/normalized_winequality_dataset.csv')

# Separar las características y la etiqueta
X = data.drop('quality', axis=1).values
y = data['quality'].values

# Dividir el dataset en train (60%), validation (20%) y test (20%)
train_size = int(0.6 * len(data))
val_size = int(0.2 * len(data))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size +
                 val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Construir el árbol de decisión
max_depth = 6  # Puedes ajustar esta profundidad
decision_tree = build_tree(X_train, y_train, max_depth=max_depth)

# Evaluación en el conjunto de validación
y_val_pred = predict_tree_batch(decision_tree, X_val)
val_accuracy, val_precision, val_recall, val_f1, val_conf_matrix = evaluate_performance(
    y_val, y_val_pred)

# Imprimir métricas de evaluación en validación
print("Evaluación en el conjunto de validación:")
print(f"Precisión (Accuracy): {val_accuracy}")
print(f"Precisión: {val_precision}")
print(f"Recall: {val_recall}")
print(f"F1 Score: {val_f1}")
print(f"Matriz de Confusión:\n{val_conf_matrix}")

# Evaluación en el conjunto de prueba
y_test_pred = predict_tree_batch(decision_tree, X_test)
test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = evaluate_performance(
    y_test, y_test_pred)

# Imprimir métricas de evaluación en prueba
print("\nEvaluación en el conjunto de prueba:")
print(f"Precisión (Accuracy): {test_accuracy}")
print(f"Precisión: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")
print(f"Matriz de Confusión:\n{test_conf_matrix}")

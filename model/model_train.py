import numpy as np
import pandas as pd

"""
Función para calcular la impureza de Gini
------------------------------------------------
Esta función se utiliza para calcular la impureza de Gini de un conjunto de etiquetas (y).
La impureza de Gini es una medida de la probabilidad de que un elemento seleccionado al azar en el conjunto sea clasificado incorrectamente si se clasifica aleatoriamente según la distribución de clases en el conjunto. Un valor de Gini de 0 indica que todas las instancias pertenecen a una sola clase, lo cual implica pureza total. Esta métrica es utilizada frecuentemente en la construcción de árboles de decisión para evaluar la calidad de una partición de los datos.
"""


def gini_impurity(y):
    # Identificamos todas las clases únicas presentes en el conjunto de etiquetas (y).
    classes = np.unique(y)

    # Inicializamos la impureza de Gini como 1.0, que representa la máxima incertidumbre.
    gini = 1.0

    # Calculamos la probabilidad de cada clase en el conjunto (y) y ajustamos el valor de Gini en consecuencia.
    for c in classes:
        # Probabilidad de que una instancia pertenezca a la clase 'c'
        p = np.sum(y == c) / len(y)

        # Se resta el cuadrado de la probabilidad de la clase al valor de Gini, lo que reduce la impureza a medida que las probabilidades de las clases se alejan de una distribución uniforme.
        gini -= p ** 2

    # Retornamos el valor calculado de la impureza de Gini para el conjunto de etiquetas dado.
    return gini


"""
Función para calcular la ganancia de información
------------------------------------------------
La ganancia de información es una métrica utilizada para decidir la mejor característica para dividir los datos en un nodo de un árbol de decisión. Se basa en la reducción de la impureza de Gini como resultado de la partición de los datos. La ganancia de información mide la diferencia entre la impureza de Gini antes de la partición y la impureza ponderada de Gini después de la partición.
Cuanto mayor sea la ganancia de información, mejor será la partición en términos de pureza.

"""


def information_gain(y, left_y, right_y):
    # Calculamos la proporción del conjunto de datos que va a la rama izquierda tras la partición.
    p = len(left_y) / len(y)
    """
    Calculamos la ganancia de información restando la impureza de Gini ponderada (después de la partición) de la impureza de Gini original (antes de la partición).

    La primera parte de la fórmula representa la impureza original, y la segunda parte resta la impureza ponderada después de la partición en ramas izquierda y derecha.
    """
    return gini_impurity(y) - (p * gini_impurity(left_y) + (1 - p) * gini_impurity(right_y))


"""
Esta función busca la mejor división de un conjunto de datos basado en la ganancia de información.
Para cada característica en X, evalúa todos los valores posibles y calcula cómo dividir los datos en dos subconjuntos (izquierda y derecha) utilizando cada valor como punto de corte. 
La función mide la ganancia de información para cada posible división, comparando la impureza de Gini antes y después de la división. El objetivo es maximizar esta ganancia, identificando la columna y el valor que proporcionan la mejor separación de los datos. La función devuelve tanto la mayor ganancia encontrada como los detalles de la división correspondiente.

"""


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


"""
Esta función construye un árbol de decisión recursivamente, dividiendo los datos en subconjuntos más pequeños en cada paso, basándose en la mejor ganancia de información obtenida. 

El proceso de división continúa hasta que se alcanza la profundidad máxima del árbol o todos los elementos en un nodo pertenecen a la misma clase. En cada nodo, se selecciona la mejor división usando la función best_split. Si no hay ganancia de información, o si se alcanza un nodo puro o la profundidad máxima, la función retorna la clase mayoritaria en ese nodo. De lo contrario, la función construye las ramas izquierda y derecha del árbol y las une en un nodo que describe la división seleccionada.

"""


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


"""
Esta función recursiva realiza predicciones utilizando un árbol de decisión previamente construido. 

Dado un árbol y un conjunto de características X, la función navega por el árbol desde la raíz hasta un nodo hoja, siguiendo las decisiones en cada nodo basadas en el valor de la característica correspondiente. 

Si el nodo actual es un diccionario (es decir, no es un nodo hoja), la función selecciona la rama izquierda o derecha dependiendo del valor de X en la columna indicada. Una vez que se alcanza un nodo hoja, la función retorna la clase predicha.

"""


def predict_tree(tree, X):
    if isinstance(tree, dict):
        if X[tree['column']] <= tree['value']:
            return predict_tree(tree['left'], X)
        else:
            return predict_tree(tree['right'], X)
    else:
        return tree


"""
Esta función aplica el árbol de decisión para hacer predicciones en un conjunto completo de datos X. 

Utiliza la función predict_tree para predecir la clase de cada fila (instancia) en X, y devuelve un arreglo de NumPy con todas las predicciones. Este enfoque permite procesar múltiples instancias en una sola llamada de función, lo que es útil para hacer predicciones en lotes de datos de manera eficiente.

"""


def predict_tree_batch(tree, X):
    return np.array([predict_tree(tree, row) for row in X])


"""

Esta función evalúa el rendimiento de un modelo de clasificación calculando varias métricas clave: exactitud (accuracy), precisión, recall, y F1-score. La exactitud se calcula como la proporción de etiquetas correctamente predichas. La precisión y el recall se calculan para evitar errores de división por cero, y el F1-score combina estas dos métricas para proporcionar una única medida del rendimiento. La función también genera una matriz de confusión, que muestra la distribución de las predicciones en comparación con las etiquetas reales. Estos resultados proporcionan una visión integral del rendimiento del modelo en un conjunto de datos de prueba.

"""


def evaluate_performance(actual_labels, predicted_labels):
    accuracy = np.mean(actual_labels == predicted_labels)

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


# Aquí se carga el dataset de un archivo CSV con ayuda de pandas
data = pd.read_csv(
    '/Users/aguero/Desktop/A01754412---Portafolio-Implementaci-n/Data/normalized_winequality_dataset.csv')

# Se separan las variables predictoras (X) de la variable objetivo (y), que es la calidad del vino
X = data.drop('quality', axis=1).values
y = data['quality'].values

# Dividir el dataset en train (60%), validation (20%) y test (20%) que vendrian siendo los tamaños de los conjuntos de entrenamiento, validación y prueba

train_size = int(0.6 * len(data))
val_size = int(0.2 * len(data))

# Se crean los conjuntos de datos para entrenamiento, validación y prueba
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size +
                 val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Se construye un árbol de decisión usando los datos de entrenamiento con una profundidad máxima especificada
max_depth = 6  # NOTA: Aqui podemos ajustar la profundidad que tendra nuestro arbol
decision_tree = build_tree(X_train, y_train, max_depth=max_depth)

# Evaluación en el conjunto de validación
# Se hacen predicciones en el conjunto de validación usando el árbol de decisión
y_val_pred = predict_tree_batch(decision_tree, X_val)

# Se calculan las métricas de rendimiento en el conjunto de validación: accuracy, precision, recall, F1 score, y la matriz de confusión
val_accuracy, val_precision, val_recall, val_f1, val_conf_matrix = evaluate_performance(
    y_val, y_val_pred)

"""

Imprimir métricas de evaluación en validación
---------------------------------------------
- Precisión (Accuracy): Mide el porcentaje de predicciones correctas sobre el total de predicciones. 
Es una métrica global que refleja qué tan bien está funcionando el modelo en general.

- Precisión (Precision): Mide la proporción de verdaderos positivos entre todas las instancias que fueron predichas como positivas. Es crucial en escenarios donde los falsos positivos tienen un alto costo.

- Recall: Mide la proporción de verdaderos positivos entre todas las instancias que realmente son positivas.
Es importante cuando es crítico identificar todos los positivos, como en la detección de enfermedades.

- F1 Score: Es la media armónica entre la precisión y el recall, proporcionando un balance entre ambas métricas.
Es útil en situaciones donde se necesita un equilibrio entre evitar falsos positivos y falsos negativos.

- Matriz de Confusión: Es una tabla que muestra el número de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos. Ayuda a visualizar cómo se distribuyen las predicciones del modelo.

"""

print("Evaluación en el conjunto de validación:")
print(f"Precisión (Accuracy): {val_accuracy}")
print(f"Precisión: {val_precision}")
print(f"Recall: {val_recall}")
print(f"F1 Score: {val_f1}")
print(f"Matriz de Confusión:\n{val_conf_matrix}")


"""

Se hacen predicciones en el conjunto de prueba y se calculan las métricas de rendimiento de manera similar a la validación

Esto es fundamental para entender cómo se desempeña el modelo en datos completamente nuevos y no vistos durante el entrenamiento.

"""
y_test_pred = predict_tree_batch(decision_tree, X_test)
test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = evaluate_performance(
    y_test, y_test_pred)

# Se imprimen las métricas para el conjunto de prueba
print("\nEvaluación en el conjunto de prueba:")
print(f"Precisión (Accuracy): {test_accuracy}")
print(f"Precisión: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")
print(f"Matriz de Confusión:\n{test_conf_matrix}")

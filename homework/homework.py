# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
import gzip
import pickle
import json
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

def load_and_preprocess():
    # Cargar los datos
    df_train = pd.read_csv("files/input/train_data.csv.zip", compression='zip')
    df_test = pd.read_csv("files/input/test_data.csv.zip", compression='zip')

    # Renombrar la columna objetivo
    df_train = df_train.rename(columns={"default payment next month": "default"})
    df_test = df_test.rename(columns={"default payment next month": "default"})

    # Remover la columna ID
    df_train = df_train.drop(columns=["ID"])
    df_test = df_test.drop(columns=["ID"])

    # Agrupar EDUCATION > 4 en la categoría "others" (valor 5)
    df_train.loc[df_train["EDUCATION"] > 4, "EDUCATION"] = 5
    df_test.loc[df_test["EDUCATION"] > 4, "EDUCATION"] = 5

    # Eliminar registros con información no disponible (valores NA)
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    return df_train, df_test
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

def split_features_target(df_train, df_test):
    x_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]
    x_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]
    return x_train, y_train, x_test, y_test

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
def build_pipeline(x_train):
    categorical_features = x_train.select_dtypes(include=["object", "category", "int64"]).columns[
        x_train.nunique() < 10
    ].tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    return pipeline
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {#"classifier__n_estimators": [1]
        "classifier__n_estimators": [100,200],
        "classifier__max_depth": [None],
        "classifier__max_samples": [0.6, 0.8],
        "classifier__min_samples_split": [2, 5],
        "classifier__max_features": ['sqrt','log2',5,8],
    }
    scorer = make_scorer(balanced_accuracy_score)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring=scorer,
        n_jobs=-1,
        verbose=4
    )
    grid_search.fit(x_train, y_train)
    return grid_search, grid_search.best_params_

# Ejemplo de uso:
# pipeline = build_pipeline(x_train)
# best_model, best_params = optimize_hyperparameters(pipeline, x_train, y_train)
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
def save_model(model, path="files/models/model.pkl.gz"):
    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Guardar el modelo
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo guardado en {path}")
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
import os
import json
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def calcular_metricas(modelo, X_train, y_train, X_test, y_test, ruta_salida='files/output/metrics.json'):
    # Predecir
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Función auxiliar para calcular métricas
    def calcular_metricas(y_true, y_pred, dataset_nombre):
        return {
            'type': 'metrics',
            'dataset': dataset_nombre,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

    # Calcular métricas
    metricas = [
        calcular_metricas(y_train, y_pred_train, 'train'),
        calcular_metricas(y_test, y_pred_test, 'test')
    ]

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    # Guardar en formato JSONL (una línea por métrica)
    with open(ruta_salida, 'w', encoding='utf-8') as f:  # w para sobrescribir
        for entrada in metricas:
            json.dump(entrada, f)
            f.write('\n')

    print(f"Métricas guardadas en {ruta_salida}")


def agregar_matrices_confusion(modelo, X_train, y_train, X_test, y_test, ruta_salida='files/output/metrics.json'):
    # Predecir
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Función para convertir matriz de confusión al formato requerido
    def formato_cm(y_true, y_pred, dataset_nombre):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            'type': 'cm_matrix',
            'dataset': dataset_nombre,
            'true_0': {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            'true_1': {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
        }

    matrices = [
        formato_cm(y_train, y_pred_train, 'train'),
        formato_cm(y_test, y_pred_test, 'test')
    ]

    # Guardar en modo append, una línea por matriz
    with open(ruta_salida, 'a', encoding='utf-8') as f:
        for entrada in matrices:
            json.dump(entrada, f)
            f.write('\n')

    print(f"Matrices de confusión agregadas a {ruta_salida}")

df_train, df_test = load_and_preprocess()

X_train, y_train, X_test, y_test = split_features_target(df_train, df_test)

pipeline = build_pipeline(X_train)

gridsearch, best_params = optimize_hyperparameters(pipeline, X_train, y_train)

# Ajustar el pipeline con los mejores hiperparámetros
gridsearch.estimator.set_params(**best_params)
gridsearch.estimator.fit(X_train, y_train)
save_model(gridsearch)

calcular_metricas(pipeline, X_train, y_train, X_test, y_test)

agregar_matrices_confusion(pipeline, X_train, y_train, X_test, y_test)
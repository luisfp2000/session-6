import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: Modify this list to include the numerical columns
NUMERICAL_VARS = ['pclass','survived','age','sibsp','parch','fare']

# Crear custom transformer


class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Crear una copia del DataFrame X para evitar modificar los datos originales
        X_transformed = X.copy()

        # Para cada variable en la lista de variables
        for variable in self.variables:
            # Crear una nueva columna con el nombre "variable_missing" que será 1 si hay valores faltantes y 0 si no
            X_transformed[variable + '_nan'] = X_transformed[variable].isnull().astype(int)

        return X_transformed

# Leer el csv sin aplicar transformaciones
df = pd.read_csv("C:/Users/luis.fernandez.COPPEL/LFPGit/session-6/activity/raw-data.csv")

# Imprimir los primeros datos
print(df.head(10))

mi = MissingIndicator(variables=NUMERICAL_VARS)
# Aplicar las transformaciones
df_mi = mi.transform(df)

# Imprimir resultados despues de las transformaciones
print(df_mi.head(20))

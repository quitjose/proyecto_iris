from sklearn import datasets
import pandas as pd

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()

# Crear un DataFrame de pandas para facilitar la visualización
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Agregar la columna de etiquetas de clase al DataFrame
iris_df['target'] = iris.target

# Imprimir las primeras filas del DataFrame
print(iris_df.head().T)

# Exportar el DataFrame a un archivo CSV
#iris_df.to_csv('iris_dataset.csv', index=False)

#print("DataFrame exportado a 'iris_dataset.csv'")


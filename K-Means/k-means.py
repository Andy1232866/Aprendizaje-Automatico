# ----------------- IMPORTS -----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# ----------------- CARGA DATOS -----------------
dataframe = pd.read_csv(r'analisis.csv')

print('\n------------------------------------------------------------------------------------\n')
print(dataframe.head())
print('\n------------------------------------------------------------------------------------\n')
print(dataframe.describe())
print('\n------------------------------------------------------------------------------------\n')
print(dataframe.groupby('categoria').size())

# Histograma (sin categoria)
dataframe.drop(['categoria'], axis=1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='categoria', height=4, vars=['op', 'ex', 'ag'], kind='scatter')

# ----------------- NORMALIZACIÓN -----------------
features = ['op', 'ex', 'ag']
X = np.array(dataframe[features])
y = np.array(dataframe['categoria'])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)  # <<< Todo desde aquí normalizado >>>

print('\n\n', X_norm.shape, '\n')

# ----------------- GRÁFICA 3D NORMALIZADA -----------------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colores = ['blue', 'red', 'green', 'blue', 'cyan', 'yellow', 'orange', 'black', 'pink', 'brown', 'purple']
asignar = [colores[row] for row in y]
ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], c=asignar, s=60)

# ----------------- ELBOW -----------------
Nc = range(1, 20)
kmeans_models = [KMeans(n_clusters=i, n_init=20, random_state=42).fit(X_norm) for i in Nc]
inertia = [model.inertia_ for model in kmeans_models]

plt.figure()
plt.plot(Nc, inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# ----------------- KMEANS NORMALIZADO -----------------
kmeans = KMeans(n_clusters=5, n_init=20, random_state=42).fit(X_norm)
centroids = kmeans.cluster_centers_
print('\n------------------------------------------------------------------------------------\n')
print('Centroides Normalizados\n')
print(centroids)

labels = kmeans.labels_
C = centroids

colores = ['red', 'green', 'blue', 'cyan', 'yellow']
asignar = [colores[label] for label in labels]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], c=asignar, s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

# ----------------- PROYECCIÓN 2D -----------------
plt.figure()
f1 = X_norm[:, 0]
f2 = X_norm[:, 1]
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()

# ----------------- CONTEO POR GRUPO -----------------
copy = pd.DataFrame()
copy['usuario'] = dataframe['usuario'].values
copy['categoria'] = dataframe['categoria'].values
copy['label'] = labels

cantidadGrupo = pd.DataFrame()
cantidadGrupo['color'] = colores
cantidadGrupo['cantidad'] = copy.groupby('label').size()
print('\n------------------------------------------------------------------------------------\n')
print('Usuarios \n')
print(cantidadGrupo)

# Ejemplo grupo 0
group_referrer_index = copy['label'] == 0
group_referrals = copy[group_referrer_index]

diversidadGrupo = pd.DataFrame()
diversidadGrupo['categoria'] = [0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad'] = group_referrals.groupby('categoria').size()
print('\n------------------------------------------------------------------------------------\n')
print('Clusters \n')
print(diversidadGrupo, '\n')

# Representantes de cada cluster
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_norm)
print('\n------------------------------------------------------------------------------------\n')
print('Índice Usuarios Representativos:', closest, '\n')

users = dataframe['usuario'].values
for row in closest:
    print('Índice:', row, 'Usuario:', users[row])

# Mostrar usuarios cluster 0
print('\nPertenecientes Cluster 0 (Rojo)\n')
for index, row in copy.iterrows():
    if row['label'] == 0:
        print("{:<20} {:<5} {:<0}".format(row['usuario'], row['categoria'], row['label']))

# ----------------- CLASIFICACIÓN NUEVO REGISTRO NORMALIZADO -----------------
X_new = np.array([[45.92, 57.74, 15.66]])  # datos crudos
X_new_norm = scaler.transform(X_new)       # se normalizan con el mismo scaler
new_labels = kmeans.predict(X_new_norm)
print('\nClúster Asignado (Normalizado):', new_labels)



# ----------------- RESUMEN CARACTERÍSTICA DOMINANTE -----------------
df_centroids = pd.DataFrame(centroids, columns = features)
df_centroids.index.name = 'Cluster'

predominant_feature = df_centroids.abs().idxmax(axis = 1)

df_summary = pd.DataFrame(predominant_feature, columns=['Caracteristica_Predominante'])
df_summary['Valor_del_Centroide'] = df_summary.apply(
    lambda row: df_centroids.loc[row.name, row['Caracteristica_Predominante']], axis = 1)
df_summary['Tipo'] = np.where(df_summary['Valor_del_Centroide'] > 0, 'Alta', 'Baja')

print('\n------------------------------------------------------------------------------------')
print('         Resumen: Característica que más define a cada Clúster')
print('------------------------------------------------------------------------------------\n')
print(df_summary)


# “op” = Openness to experience – grado de apertura mental a nuevas experiencias, curiosidad, arte
# “ex” = Extraversion – grado de timidez, solitario o participación ante el grupo social
# “ag” = Agreeableness – grado de empatía con los demás, temperamento
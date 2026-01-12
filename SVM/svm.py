import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = (8, 6)

df = pd.read_csv('breast_cancer.csv')
print('-' * 70, '\n')
print(df.head())
print('\n', '-' * 70)

X = df.drop('class', axis = 1)
y = df['class']

y_conv = pd.Series(y).replace({'benign': 0, 'malignant': 1, 2: 0, 4: 1})

plt.hist(y_conv.map({0: 'Benigno', 1: 'Maligno'}), bins = 3)
plt.title('Distribución de Clases (Benigno vs. Maligno)')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.show()

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_conv, test_size = 0.2, random_state = 42)

clf = svm.SVC(kernel ='rbf')
clf.fit(X_pca, y_conv)

y_pred = clf.predict(X_test)
exa = accuracy_score(y_test, y_pred)
print(f'\nExactitud del modelo: {exa:.4f}')
print('\n', '-' * 70)

# Matriz de confusión
mc = confusion_matrix(y_test, y_pred)

sb.heatmap(mc, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Benigno (Pred)', 'Maligno (Pred)'], yticklabels = ['Benigno (Real)', 'Maligno (Real)'])
plt.xlabel('Predicción del Modelo')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión del SVM')
plt.show()

benigno = {
    'clump_thickness': 3.0, 'uniformity_of_cell_size': 1.0, 'uniformity_of_cell_shape': 1.0,
    'marginal_adhesion': 1.0, 'single_epithelial_cell_size': 2.0, 'bare_nuclei': 1.0,
    'bland_chromatin': 2.0, 'normal_nucleoli': 1.0, 'mitoses': 1.0
}
maligno = {
    'clump_thickness': 8.0, 'uniformity_of_cell_size': 10.0, 'uniformity_of_cell_shape': 10.0,
    'marginal_adhesion': 8.0, 'single_epithelial_cell_size': 7.0, 'bare_nuclei': 10.0,
    'bland_chromatin': 9.0, 'normal_nucleoli': 7.0, 'mitoses': 1.0,
}

df_benigno = pd.DataFrame([benigno]).reindex(columns = X.columns, fill_value = 0)
df_maligno = pd.DataFrame([maligno]).reindex(columns = X.columns, fill_value = 0)
benigno_pca = pca.transform(df_benigno)
maligno_pca = pca.transform(df_maligno)

pred_benigno = clf.predict(benigno_pca)
pred_maligno = clf.predict(maligno_pca)
print(f'\nLa predicción es: {pred_benigno}')
print(f'La predicción es: {pred_maligno}')
# [0] es benigno y [1] es maligno
print('\n', '-' * 70)

plt.figure(figsize = (12, 10))
ax = plt.gca()

# Puntos de los datos originales
plt.scatter(X_pca[y_conv == 0, 0], X_pca[y_conv == 0, 1], c = 'blue', label = 'Benigno', edgecolors = 'k')
plt.scatter(X_pca[y_conv == 1, 0], X_pca[y_conv == 1, 1], c = 'red', label = 'Maligno', edgecolors = 'k')

# Puntos de los datos de predicción
plt.scatter(benigno_pca[:, 0], benigno_pca[:, 1], s = 250, c = 'green', marker = '*', label = 'Nuevo Benigno', edgecolors = 'k')
plt.scatter(maligno_pca[:, 0], maligno_pca[:, 1], s = 250, c = 'red', marker = '*', label = 'Nuevo Maligno', edgecolors = 'k')

# Vectores de soporte
sv = clf.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s = 180, facecolors = 'none', edgecolors = 'k', label = 'Vectores de soporte')

# Hiperplano
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                       np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contour(xx, yy, Z, colors = 'k', levels = [-1, 0, 1], alpha = 0.8, linestyles = ['--', '-', '--'])

ax.contourf(xx, yy, Z, alpha = 0.2, cmap = 'coolwarm')

plt.title('Plano del SVM')
plt.legend()
plt.show()

print(f'\nNúmero de vectores de soporte: {len(clf.support_vectors_)}')
print('\n', '-' * 70)
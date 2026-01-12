import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Depression.csv')
print("-" * 70)
print(df.head())
print("-" * 70, '\n\n')

df['depression_label'].value_counts().plot.bar(color=['lightgreen', 'lightblue'])
plt.xticks(rotation = 75)
plt.show()

print(f'{" Conteo de Variables ":-^70}', '\n')
print(df.groupby('depression_label').size().reset_index(name = 'Cantidad'))
print("-" * 70)

X = df.drop(['university', 'current_cgpa', 'department', 'has_scholarship',
             'academic_year', 'depression_value', 'depression_label'], axis = 1)

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(df['depression_label'])

labels = {}
cat = X.select_dtypes(include='object').columns
for col in cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    labels[col] = le
    
print("\nColumnas codificadas:", list(cat), '\n')
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

clf = RandomForestClassifier(
    n_estimators = 50,
    max_depth = 6, max_leaf_nodes = 10
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("-" * 70, '\n')
print(f"Precisión del modelo: {accuracy * 100:.2f}%" '\n')
print("-" * 70)

plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd',
            cmap='Blues', xticklabels = y_encoder.classes_, yticklabels = y_encoder.classes_)
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

usuario = {
    'age': '27-30',
    'gender': 'Female',
    'q1_interest': 3,   # Sin interés total
    'q2_hopeless': 2,   # Siempre sin esperanza
    'q3_sleep': 1,      # Insomnio extremo o duerme demasiado
    'q4_energy': 4,     # Sin energía por completo
    'q5_appetite': 3,   # Cambios extremos de apetito
    'q6_self_estreem': 2, # Autoestima extremadamente baja
    'q7_concentration': 4, # Incapaz de concentrarse
    'q8_movement': 2,      # Muy agitado o lento
    'q9_self_harm': 1      # Pensamientos constantes de autolesión
}

new_df = pd.DataFrame([usuario])
for col, le in labels.items():
    new_df[col] = le.transform(new_df[col])

new_df = new_df.reindex(columns = X.columns, fill_value = 0)

prediccion = clf.predict(new_df)[0]
prediccion_label = y_encoder.inverse_transform([prediccion])[0]

print(f'\nLa predicción es: {prediccion} ({prediccion_label})')
print('\n', "-" * 70)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Charger et préparer les données
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()  # Supprime les espaces dans les noms de colonnes
    return data

# Encodage des variables catégoriques
def preprocess_data(data, categorical_columns):
    label_encoders = {}
    encoded_data = data.copy()
    for col in categorical_columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        label_encoders[col] = le
    return encoded_data, label_encoders

# Charger les données
file_path = "data_imputationfinal.xlsx"  # Remplacez par votre fichier
data = load_data(file_path)

# Colonnes pour chaque tâche
columns_task1 = ['Numero IMPUTATION', 'Numero DOSSIER', 'ORIGINE/AFFAIRES', 'OBJET', 'JURIDICTIONS', 'INSTRUCTIONS ET DELAIS']
columns_task2 = ['complexité', 'domaine_juridique', 'urgence']

# Ajouter des colonnes par défaut pour la tâche 2
default_values_task2 = {
    'complexité': 'simple',
    'domaine_juridique': 'Droit civil',
    'urgence': 'Moyenne'
}
for col, default in default_values_task2.items():
    data[col] = default

# Prétraitement des données
encoded_data, label_encoders = preprocess_data(data, columns_task1 + columns_task2 + ['juriste'])

# Tâche 1 : Prédiction du juriste
X_task1 = encoded_data[columns_task1]
y_task1 = encoded_data['juriste']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_task1, y_task1, test_size=0.1, random_state=50)
model_task1 = RandomForestClassifier(random_state=50)
model_task1.fit(X_train1, y_train1)
accuracy_task1 = accuracy_score(y_test1, model_task1.predict(X_test1))

# Tâche 2 : Prédiction du numéro de dossier
X_task2 = encoded_data[columns_task1 + columns_task2]
y_task2 = encoded_data['Numero DOSSIER']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_task2, y_task2, test_size=0.1, random_state=50)
model_task2 = RandomForestClassifier(random_state=50)
model_task2.fit(X_train2, y_train2)
accuracy_task2 = accuracy_score(y_test2, model_task2.predict(X_test2))

# Interface utilisateur
st.title("📊 Prédictions Juridiques avec Machine Learning")
st.sidebar.header("Performance des Modèles")
st.sidebar.write(f"Précision Tâche 1 (Juriste) : **{accuracy_task1 * 100:.2f}%**")
st.sidebar.write(f"Précision Tâche 2 (Numéro de Dossier) : **{accuracy_task2 * 100:.2f}%**")

# Navigation
task = st.radio("Choisissez une tâche :", ["Prédiction du Juriste", "Prédiction du Numéro de Dossier"])

if task == "Prédiction du Juriste":
    st.header("🔍 Prédiction du Juriste")
    user_inputs = {col: st.selectbox(col, label_encoders[col].classes_) for col in columns_task1}
    if st.button("🔮 Prédire"):
        input_data = {col: label_encoders[col].transform([user_inputs[col]])[0] for col in columns_task1}
        input_df = pd.DataFrame([input_data])
        prediction = model_task1.predict(input_df)[0]
        predicted_juriste = label_encoders['juriste'].inverse_transform([prediction])[0]
        st.success(f"🧑‍⚖️ Juriste recommandé : **{predicted_juriste}**")

elif task == "Prédiction du Numéro de Dossier":
    st.header("🔍 Prédiction du Numéro de Dossier")
    user_inputs = {col: st.selectbox(col, label_encoders[col].classes_) for col in columns_task1 + columns_task2}
    if st.button("🔮 Prédire"):
        input_data = {col: label_encoders[col].transform([user_inputs[col]])[0] for col in columns_task1 + columns_task2}
        input_df = pd.DataFrame([input_data])
        prediction = model_task2.predict(input_df)[0]
        st.success(f"📂 Numéro de Dossier prédit : **{prediction}**")

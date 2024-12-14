import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Charger et préparer les données
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
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

# Entraînement du modèle
@st.cache_data
def train_model(X, y):
    model = RandomForestClassifier(random_state=50)
    model.fit(X, y)
    return model

# Interface utilisateur
st.title("📊 Prédictions Juridiques avec Machine Learning")

# Téléchargement du fichier
uploaded_file = st.file_uploader("Charger un fichier Excel", type=["xlsx"])
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Colonnes pour chaque tâche
    columns_task1 = ['Numero IMPUTATION', 'Numero DOSSIER', 'ORIGINE/AFFAIRES', 'OBJET', 'JURIDICTIONS', 'INSTRUCTIONS ET DELAIS']
    columns_task2 = ['complexité', 'domaine_juridique', 'urgence']

    # Validation des colonnes
    missing_columns = [col for col in columns_task1 + columns_task2 + ['juriste'] if col not in data.columns]
    if missing_columns:
        st.error(f"Colonnes manquantes dans le fichier : {', '.join(missing_columns)}")
        st.stop()

    # Ajouter des colonnes par défaut pour la tâche 2
    default_values_task2 = {
        'complexité': 'simple',
        'domaine_juridique': 'Droit civil',
        'urgence': 'Moyenne'
    }
    for col, default in default_values_task2.items():
        if col not in data.columns:
            data[col] = default

    # Prétraitement des données
    encoded_data, label_encoders = preprocess_data(data, columns_task1 + columns_task2 + ['juriste'])

    # Tâche 1 : Prédiction du juriste
    X_task1 = encoded_data[columns_task1]
    y_task1 = encoded_data['juriste']
    model_task1 = train_model(X_task1, y_task1)
    accuracy_task1 = accuracy_score(y_task1, model_task1.predict(X_task1))

    # Tâche 2 : Prédiction du numéro de dossier
    X_task2 = encoded_data[columns_task1 + columns_task2]
    y_task2 = encoded_data['Numero DOSSIER']
    model_task2 = train_model(X_task2, y_task2)
    accuracy_task2 = accuracy_score(y_task2, model_task2.predict(X_task2))

    # Sidebar : Performances des modèles
    st.sidebar.header("Performance des Modèles")
    st.sidebar.write(f"Précision Tâche 1 (Juriste) : **{accuracy_task1 * 100:.2f}%**")
    st.sidebar.write(f"Précision Tâche 2 (Numéro de Dossier) : **{accuracy_task2 * 100:.2f}%**")

    # Navigation
    task = st.radio("Choisissez une tâche :", ["Prédiction du Juriste", "Prédiction du Numéro de Dossier"])

    if task == "Prédiction du Juriste":
        st.header("🔍 Prédiction du Juriste")
        user_inputs = {col: st.selectbox(col, label_encoders[col].classes_) for col in columns_task1}
        if st.button("🔮 Prédire"):
            try:
                input_data = {col: label_encoders[col].transform([user_inputs[col]])[0] for col in columns_task1}
                input_df = pd.DataFrame([input_data])
                prediction = model_task1.predict(input_df)[0]
                predicted_juriste = label_encoders['juriste'].inverse_transform([prediction])[0]
                st.success(f"🧑‍⚖️ Juriste recommandé : **{predicted_juriste}**")
            except KeyError as e:
                st.error(f"Valeur inattendue pour {e.args[0]}. Veuillez vérifier vos entrées.")

    elif task == "Prédiction du Numéro de Dossier":
        st.header("🔍 Prédiction du Numéro de Dossier")
        user_inputs = {col: st.selectbox(col, label_encoders[col].classes_) for col in columns_task1 + columns_task2}
        if st.button("🔮 Prédire"):
            try:
                input_data = {col: label_encoders[col].transform([user_inputs[col]])[0] for col in columns_task1 + columns_task2}
                input_df = pd.DataFrame([input_data])
                prediction = model_task2.predict(input_df)[0]
                st.success(f"📂 Numéro de Dossier prédit : **{prediction}**")
            except KeyError as e:
                st.error(f"Valeur inattendue pour {e.args[0]}. Veuillez vérifier vos entrées.")

else:
    st.warning("Veuillez charger un fichier Excel pour commencer.")

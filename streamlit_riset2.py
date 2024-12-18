import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from bayes_opt import BayesianOptimization
from numbers import Real


# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Input Data", "Preprocessing", "Splitting Data", "Klasifikasi", "Prediksi"])

data = None

if menu == "Input Data":
    st.header("Input Data")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Uploaded:")
        st.dataframe(data)

        # Option to drop columns
        if st.checkbox("Drop Columns"):
            columns_to_drop = st.multiselect("Select columns to drop", data.columns)
            if st.button("Drop Selected Columns"):
                data = data.drop(columns=columns_to_drop)
                st.write("Updated Data:")
                st.dataframe(data)

        st.session_state.data = data

elif menu == "Preprocessing":
    st.header("Preprocessing")

    if "data" in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        st.write("Original Data:")
        st.dataframe(data)

        # Remove duplicates
        if st.checkbox("Remove Duplicates"):
            data = data.drop_duplicates()
            st.write("Data after removing duplicates:")
            st.dataframe(data)

        # Transform categorical data
        if st.checkbox("Transform Categorical Columns"):
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            st.write("Data after transforming categorical columns:")
            st.dataframe(data)

        # Balancing data menggunakan ADASYN
        if st.checkbox("Balance Data (ADASYN)"):
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Periksa keseimbangan data sebelum menjalankan ADASYN
            data_counts = data.iloc[:, -1].value_counts().sort_values(ascending=False)
            st.write("Distribusi kelas sebelum balancing:")
            st.write(data_counts)

            adasyn = ADASYN(sampling_strategy='minority')  # Atur sampling_strategy ke 'minority'
            X, y = adasyn.fit_resample(X, y)
            data = pd.concat([pd.DataFrame(X, columns=data.columns[:-1]), pd.Series(y, name=data.columns[-1])], axis=1)
            st.write("Data setelah balancing:")
            st.dataframe(data)
            
            data_counts = data.iloc[:, -1].value_counts().sort_values(ascending=False)
            st.write("Distribusi kelas setelah balancing:")
            st.write(data_counts)

        st.session_state.data = data
        
    else:
        st.warning("Please upload and process the data in the Input Data menu first.")

elif menu == "Splitting Data":
    st.header("Splitting Data")

    if "data" in st.session_state and st.session_state.data is not None:
        data = st.session_state.data

        split_ratio = st.selectbox("Select train-test split ratio", ["80:20", "70:30", "60:40"])

        if split_ratio == "80:20":
            test_size = 0.2
        elif split_ratio == "70:30":
            test_size = 0.3
        else:
            test_size = 0.4

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        st.write("Train Data:")
        st.dataframe(pd.concat([X_train, y_train], axis=1))

        st.write("Test Data:")
        st.dataframe(pd.concat([X_test, y_test], axis=1))

        st.session_state.split_data = (X_train, X_test, y_train, y_test)
    else:
        st.warning("Please preprocess the data first.")

elif menu == "Klasifikasi":
    st.header("Klasifikasi")

    if "split_data" in st.session_state and st.session_state.split_data is not None:
        X_train, X_test, y_train, y_test = st.session_state.split_data

        st.write("Train and Test Data are ready.")

        # Define XGBoost and Bayesian Optimization
        if st.button("Modeling"):

            def xgb_evaluate(max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
                model = XGBClassifier(
                    max_depth=int(max_depth),
                    learning_rate=learning_rate,
                    n_estimators=int(n_estimators),
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    eval_metric='logloss'
                )
                scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
                return scores.mean()
            
            param_bounds = {
                'colsample_bytree': (0.7, 1.0),
                'learning_rate': (0.01, 1.0),
                'max_depth': (70, 100),
                'n_estimators': (10, 5000),
                'subsample': (0.5, 1.0),
            }

            optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=param_bounds, random_state=42)
            optimizer.maximize(init_points=5, n_iter=10)

            st.write("Best Parameters Found:")
            best_params = optimizer.max['params']
            best_params['colsample_bytree'] = best_params['colsample_bytree']
            best_params['learning_rate'] = best_params['learning_rate']
            best_params['max_depth'] = int(best_params['max_depth'])
            best_params['n_estimators'] = int(best_params['n_estimators'])
            best_params['subsample'] = best_params['subsample']
            st.json(best_params)

            # 6. Training Model
            st.subheader("Training XGBoost Model")
            final_model = XGBClassifier(
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                n_estimators=best_params['n_estimators'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
                random_state=42,
                eval_metric='logloss'
            )
            final_model.fit(X_train, y_train)
            
            # Simpan model ke file
            joblib.dump(final_model, 'xgboost_model.pkl')
            st.success("Model telah disimpan sebagai 'xgboost_model.pkl'.")

            # 7. Evaluasi Model
            st.subheader("Evaluasi Model")
            y_pred = final_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            classification = classification_report(y_test, y_pred)

            st.write(f"**Accuracy:** {accuracy:.4f}")
            
            # Confusion Matrix
            st.write("Confusion Matrix:")
            st.text(conf_matrix)
            
            # Classification Report
            st.write("Classification Report:")
            st.text(classification)
    else:
        st.warning("Please split the data first.")

elif menu == "Prediksi":
    st.header("Prediksi Menggunakan Model yang Telah Disimpan")

    # Muat model
    try:
        model = joblib.load('xgboost_model.pkl')
        st.success("Model berhasil dimuat.")
    except FileNotFoundError:
        st.error("Model belum tersedia. Harap lakukan klasifikasi terlebih dahulu.")
        st.stop()

    # Input fitur untuk prediksi
    st.subheader("Masukkan Nilai Fitur:")
    gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
    age = st.number_input("Age:", min_value=0, max_value=120)
    HbA1c_level = st.number_input("HbA1c Level:", min_value=0.0, max_value=10.0)
    blood_glucose_level = st.number_input("Blood Glucose Level:", min_value=0.0, max_value=500.0)

    # Tombol untuk prediksi
    if st.button("Prediksi"):
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        # Prediksi
        prediction = model.predict(input_data)[0]
        st.write(f"**Hasil Prediksi:** {'Diabetes' if prediction == 1 else 'Tidak Diabetes'}")

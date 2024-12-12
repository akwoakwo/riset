import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Input Data", "Preprocessing", "Splitting Data", "Klasifikasi"])

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
    
        # st.session_state.data = data
    
    else:
        st.warning("Please preprocess the data first.")

elif menu == "Klasifikasi":
    st.header("Klasifikasi")

    if "split_data" in st.session_state and st.session_state.split_data is not None:
        X_train, X_test, y_train, y_test = st.session_state.split_data
        
        # User Input Features
        feature_names = list(X_train.columns)
        user_input = {}
        for feature_name in feature_names:
            # Adjust input type based on data (e.g., st.number_input, st.selectbox)
            user_input[feature_name] = st.number_input(feature_name)
        
        user_data = pd.DataFrame([user_input], columns=X_train.columns)

        # st.write("Train and Test Data are ready.")

        def predict_accuracy(user_input, X_train, y_train, X_test, y_test):
            user_data = pd.DataFrame([user_input], columns=X_train.columns)
            param_grid = {
                'learning_rate': Real(0.01, 1.0, prior='uniform'),    
                'n_estimators': Integer(10, 5000),                          
                'max_depth': Integer(70, 100),                               
                'min_child_weight': Integer(10, 15),
                'subsample': Real(0.5, 1.0, prior='uniform'),
                'colsample_bytree': Real(0.7, 1.0, prior='uniform'),
                'gamma': Real(0.05, 0.1, prior='uniform'),                      
                'reg_alpha': Real(0.01, 100, prior='uniform'),
                'reg_lambda': Real(0.01, 100, prior='uniform'),
            }

            xgb = XGBClassifier(eval_metric='logloss')
            opt = BayesSearchCV(xgb, param_grid, n_iter=50, cv=5, random_state=42)
            opt.fit(X_train, y_train)
            
            y_pred = opt.predict(user_data)
            accuracy = opt.score(X_test, y_test)
            best_params = opt.best_params_

            return y_pred[0], accuracy, best_params
    
        # Define XGBoost and Bayesian Optimization
        if st.button("Prediksi"):
            prediction, accuracy, best_params = predict_accuracy(user_input, X_train, y_train, X_test, y_test)
            st.write("Prediksi:", prediction)
            st.write("Akurasi Model:", accuracy)
            st.write("Parameter Terbaik:", best_params)
    else:
        st.warning("Please split the data first.")

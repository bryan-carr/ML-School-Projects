import streamlit as st
import FinalProject

st.title("Spotify Song Genre Classifier Experiment - Decision Tree vs XGBoost")
st.write("NOTE: Recommended parameters are pre-set as default!")

st.write("##Input Decision Tree Parameters:")
tree_md = st.number_input("Decision Tree Max Depth:", min_value = 1, max_value = 50, value=15)
tree_split = st.number_input("Decision Tree min_samples_split:", min_value = 2, max_value=100, value=2)
tree_impurity = st.number_input("Decision Tree min_impurity_decrease:", min_value = 0.0, max_value = 0.5, value = 0.0)
tree_params = {
    'max_depth' : [tree_md],
    'min_samples_split' : [tree_split],
    'min_impurity_decrease' : [tree_impurity]
}

st.write("##Input XGBoost Parameters:")
xgb_md = st.number_input("XGBoost Tree Max Depth:", min_value=2, max_value=10, value=5)
xgb_eta = st.number_input("XGBoost Learning Rate:", min_value=0.01, max_value=0.5, value=0.2)
xgb_steps = st.number_input("XGBoost Number of Steps:", min_value=5, max_value=50, value=10)
xgb_gamma = st.number_input("XGBoost Gamma Value:", min_value=0.01, max_value=0.9, value=0.3)
xgb_params = {
    "max_depth": [xgb_md],
    "learning_rate": [xgb_eta],
    "num_steps": [xgb_steps],
    "gamma": [xgb_gamma]
}

result = st.button("RUN ANALYSIS - Will take a few minutes!")

if result:
    FinalProject.main_function('streamlit_results.txt', tree_params, xgb_params)
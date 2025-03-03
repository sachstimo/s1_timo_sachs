import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from churn_model import load_data, train_and_save_model

X, y, state_area = load_data()

#### ---- Streamlit Dashboard Creation ---- ####

st.title("Churn Prediction Dashboard")


with st.sidebar:
    #### Get inputs for business case (& model scoring refitting)
    st.image("DATA/logo_telco.png", width=150)
    st.title("Telephonica")
    st.write("### Business Case Assumptions")
    retention_prob = st.slider('Retention Probability', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    rev_per_cust = st.number_input('Revenue per Customer', value=60)
    marketing_cost = st.number_input('Marketing cost per customer', value=10)
    opp_cost = st.toggle('Consider opportunity cost', value=False)

    tp_profit = rev_per_cust * retention_prob - marketing_cost
    fp_loss = marketing_cost
    fn_loss = tp_profit if opp_cost else 0
    tn_profit = 0

    #### Load pre-trained model
    model_file = 'xgb_model_trained.pkl'
    if not os.path.exists(model_file):
        st.write("No existing model found. Training a new model...")
        model = train_and_save_model(X, y, tp_profit, fp_loss, fn_loss, tn_profit)
        st.write("Model trained and saved as 'best_xgb_model.pkl'.")
    else:
        model = joblib.load(model_file)

    if st.button('Retrain Model'):
        model = train_and_save_model(X, y, tp_profit, fp_loss, fn_loss, tn_profit)
        st.write("")
        st.write("Successfully retrained prediction model with new business case assumptions.")

    #### Predict churn probabilities
    churn_probabilities = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    'Churn Probability': churn_probabilities,
    'State': state_area['State'],
    'Area code': state_area['Area code']
})

#### Calculate profit for different thresholds

thresholds = np.arange(0.0, 1.0, 0.01)
profits = []

for threshold in thresholds:
    predictions = (churn_probabilities >= threshold).astype(int)
    TP = np.sum((y == 1) & (predictions == 1))
    FP = np.sum((y == 0) & (predictions == 1))
    FN = np.sum((y == 1) & (predictions == 0))
    TN = np.sum((y == 0) & (predictions == 0))
    profit = (TP * tp_profit) - (FP * fp_loss) - (FN * fn_loss) + (TN * tn_profit)
    profits.append(profit)

best_threshold = thresholds[np.argmax(profits)]

binary_predictions = (churn_probabilities >= best_threshold).astype(int)
results['Churn Prediction'] = binary_predictions

#### Calculate potential profit for each customer
results['Potential Profit'] = results['Churn Prediction'] * tp_profit - results['Churn Prediction'] * tn_profit + (1 - results['Churn Prediction']) * fn_loss
total_profit = results['Potential Profit'].sum()


#### ---- Dashboard Layout ---- ####

# Overview section with key metrics
st.subheader("Overall Business Impact")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Potential Profit", f"${total_profit:,.2f}")
with col2:
    st.metric("Optimal Threshold", f"{best_threshold:.2f}")

# Area analysis section
st.subheader("Analysis by Area")
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    # Top profitable areas
    st.write("**Top 5 Areas by Profit Potential**")
    top_areas = results.groupby('Area code')['Potential Profit'].sum().nlargest(5).reset_index()
    st.dataframe(top_areas.style.format({"Potential Profit": "${:,.2f}"}), use_container_width=True)

with col2:
    # Filters in the middle column
    st.write("**Filter Options**")
    selected_state = st.selectbox("State", options=sorted(results['State'].unique()))
    filtered_areas = results[results['State'] == selected_state]['Area code'].unique()
    selected_area = st.selectbox("Area Code", options=sorted(filtered_areas))

with col3:
    # Selected area statistics
    area_data = results[(results['State'] == selected_state) & (results['Area code'] == selected_area)]
    area_profit = area_data['Potential Profit'].sum()
    area_customers = len(area_data)
    at_risk_customers = area_data['Churn Prediction'].sum()
    
    st.write("**Selected Area Metrics**")
    st.metric("Area Profit Potential", f"${area_profit:,.2f}")
    st.metric("Total Customers", f"{area_customers}")
    st.metric("At-Risk Customers", f"{int(at_risk_customers)} ({at_risk_customers/area_customers:.1%})")

# Customer details section using full width
st.subheader(f"Customer Details - {selected_state}, Area {selected_area}")
filtered_results = results[(results['State'] == selected_state) & (results['Area code'] == selected_area)]
st.dataframe(
    filtered_results[['Churn Probability', 'Churn Prediction', 'Potential Profit']]
    .sort_values('Churn Probability', ascending=False)
    .style.format({
        "Churn Probability": "{:.2%}", 
        "Potential Profit": "${:,.2f}"
    }),
    use_container_width=True
)
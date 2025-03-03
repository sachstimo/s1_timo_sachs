import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from churn_model import load_data, train_and_save_model

X, y, state_area = load_data()

#### ---- Streamlit Dashboard Creation ---- ####

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Churn Prediction Dashboard")


with st.sidebar:
    #### Get inputs for business case (& model scoring refitting)
    st.image("DATA/logo_telco.png", width=150)
    st.title("Telephonica")
    st.write("### Business Case Assumptions")
    retention_prob = st.slider('Retention Probability [in %]', min_value=0, max_value=100, value=60, step=1) / 100
    rev_per_cust = st.number_input('Revenue per Customer', value=60)
    marketing_cost = st.number_input('Marketing cost per customer', value=10)
    opp_cost = st.toggle('Include opportunity cost of missed churners', value=False)
    if opp_cost:
        opp_cost_val = st.number_input('Opportunity cost per missed churner', value=10)
    else:
        opp_cost_val = 0

    tp_profit = rev_per_cust * retention_prob - marketing_cost
    fp_loss = marketing_cost
    fn_loss = opp_cost_val
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

def calculate_profit(y, predictions):
    TP = np.sum((y == 1) & (predictions == 1))
    FP = np.sum((y == 0) & (predictions == 1))
    FN = np.sum((y == 1) & (predictions == 0))
    TN = np.sum((y == 0) & (predictions == 0))
    profit = (TP * tp_profit) - (FP * fp_loss) - (FN * fn_loss) + (TN * tn_profit)
    return profit

thresholds = np.arange(0.0, 1.0, 0.01)
profits = []

for threshold in thresholds:
    predictions = (churn_probabilities >= threshold).astype(int)
    profit = calculate_profit(y, predictions)
    profits.append(profit)

best_threshold = thresholds[np.argmax(profits)]

binary_predictions = (churn_probabilities >= best_threshold).astype(int)
results['Churn Prediction'] = binary_predictions

#### Calculate potential profit for each customer
results['Potential Profit'] = results['Churn Prediction'] * tp_profit - results['Churn Prediction'] * fp_loss - (1 - results['Churn Prediction']) * fn_loss
total_profit = results['Potential Profit'].sum()

#### ---- Dashboard Layout ---- ####

# Overview section with key metrics
st.subheader("**Overall Business Impact**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total identified churners", f"{np.sum(binary_predictions)}")
with col2:
    st.metric("Total Potential Profit", f"${total_profit:,.2f}")
with col3:
    st.metric("Optimal Prediction Threshold", f"{best_threshold:.2f}")


# Area analysis section with top 5 areas by profit potential

st.subheader("**Churn Probability by State**")

# Calculate average churn probability by state
state_churn = results.groupby('State')['Churn Probability'].mean().reset_index()
state_profit = results.groupby('State')['Potential Profit'].sum().reset_index()

# Merge the dataframes
state_data = pd.merge(state_churn, state_profit, on='State')

# Create the choropleth map
import plotly.express as px

fig = px.choropleth(
    state_data,
    locations='State',
    locationmode='USA-states',
    color='Churn Probability',
    hover_name='State',
    hover_data={'Churn Probability': ':.2%', 'Potential Profit': ':$.2f'},
    color_continuous_scale='Reds',
    scope='usa',
    labels={'Churn Probability': 'Avg. Churn Probability'}
)

fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title="Churn Probability",
        tickformat='.0%'
    )
)

st.plotly_chart(fig, use_container_width=True)

# Keep the top 5 states table for reference
st.subheader("**Top 5 States by Profit Potential**")
top_areas = results.groupby('State')['Potential Profit'].sum().nlargest(5).reset_index()
st.dataframe(top_areas.style.format({"Potential Profit": "${:,.2f}"}), use_container_width=True)

st.subheader("Analysis by State")
# Filters in the middle column
st.write("**Filter Options**")
selected_state = st.selectbox("State", options=sorted(results['State'].unique()))
filtered_areas = results[results['State'] == selected_state]['Area code'].unique()

col1, col2, col3 = st.columns(3)

st.write("**Selected Area Metrics**")
# Selected area statistics
area_data = results[results['State'] == selected_state]
area_profit = area_data['Potential Profit'].sum()
area_customers = len(area_data)
at_risk_customers = area_data['Churn Prediction'].sum()

with col1:
    st.metric("Total Customers", f"{area_customers}")
with col2:
    st.metric("At-Risk Customers", f"{int(at_risk_customers)} ({at_risk_customers/area_customers:.1%})")
with col3:    
    st.metric("Area Profit Potential", f"${area_profit:,.2f}")

# Customer details section using full width
st.subheader(f"Customer Details - {selected_state}")
filtered_results = results[results['State'] == selected_state]
st.dataframe(
    filtered_results[['Churn Probability', 'Churn Prediction']]
    .sort_values('Churn Probability', ascending=False)
    .style.format({
        "Churn Probability": "{:.2%}", 
        "Potential Profit": "${:,.2f}"
    }),
    use_container_width=True
)
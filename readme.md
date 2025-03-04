# Churn Prediction Dashboard

A data-driven solution for telecom customer churn prediction and management.

## Business Value

- **Proactive Churn Prevention**: Identify at-risk customers before they leave
- **Geospatial Analysis**: Visualize churn probabilities across different states
- **Profit Optimization**: Balance retention costs against customer value
- **Automated Alerts**: Email notifications for 10 highest-risk customers
- **Business Case Flexibility**: Adjust model parameters (e.g. marketing spent and retention rate) based on changing business conditions

## Features

- Interactive choropleth map showing churn probability by state
- State-level analysis of at-risk customers based on dynamic filters
- Configurable business parameters (retention probability, revenue, marketing costs)
- Optimized prediction threshold based on business case
- Email subscription service for churn alerts
- Performance metrics to evaluate model effectiveness

## Project Structure

- **app.py**: Main Streamlit application with dashboard interface
- **churn_model.py**: Contains model training and data loading functions
- **email_alert.py**: Email notification functionality
- **DATA/**: Contains the dataset and company logo
- **xgb_model_trained.pkl**: Saved trained model (output from churn_model.py)

## Setup Instructions

1. **Environment Setup**:
    ```
    pip install streamlit pandas numpy joblib scikit-learn plotly xgboost python-dotenv
    ```
    or

    ```
    pip install -r requirements.txt
    ```

2. **Environment Variables**:
   Create a `.env` file with the following variables:
   ```
   SENDER_EMAIL=your_email@example.com
   AUTH_PASSWORD=your_email_password
   SMTP_SERVER=your_smtp_server
   ```

3. **Running the Application**:
   ```
   streamlit run app.py
   ```

4. **First Run**:
   - The application will train a new model if none exists (but not perform the full gridsearch since hyperparameters are already set based on experience from other project)
   - Adjust business parameters in the sidebar as needed
   - Navigate through the dashboard using the sidebar links

## Limitations

- **Email Authentication**: Requires proper SMTP setup and credentials
- **Model Retraining**: Full retraining on parameter changes may cause latency
- **Static Dataset**: Currently uses pre-loaded data without real-time updates (would need to be updated for production use to load unseen data from production database in the same format)
- **Geospatial Granularity**: Analysis limited to state level, not more granular regions
- **Missing customer id or name**: Dataset does not contain a name or customer id, so for real-word use contacting the customers would be impossible

## Future Enhancements

- Real-time data integration
- More granular geographic analysis
- Include (or make up) customer names
import streamlit as st
import pandas as pd
import lightgbm as lgb


# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return lgb.Booster(model_file='model.txt')


# Predict function
def predict_single_instance(model, feature_values):
    expected_features = ['CNT_CHILDREN', 'Annual_Income', 'Loan_Amount', 'Loan_Term', 'Employment_Years',
                         'Age', 'CODE_GENDER_encoded', 'NAME_EDUCATION_TYPE_encoded', 'FLAG_OWN_CAR_encoded',
                         'FLAG_OWN_REALTY_encoded', 'NEW_EXT_SOURCE', 'PAYMENT_RATE']
    input_df = pd.DataFrame([feature_values], columns=expected_features)
    pred_prob = model.predict(input_df, num_iteration=model.best_iteration)
    return "Low Risk" if pred_prob[0] < 0.2 else "High Risk"


# Streamlit app
def main():
    st.set_page_config(layout="wide")  # Set the layout to wide mode
    col1, col2 = st.columns([3, 1])  # Create two columns, col2 will be narrower

    with col1:
        st.title('Loan Risk Prediction App')  # Example title

    with col2:
        st.image("logo.png", width=200)

    # Model loading
    model = load_model()

    # User inputs
    st.sidebar.header('Input Features')
    cnt_children = st.sidebar.number_input('Number of Children', min_value=0, max_value=20, value=0)
    annual_income = st.sidebar.number_input('Annual Income', min_value=0, value=50000)
    loan_amount = st.sidebar.number_input('Loan Amount', min_value=0, value=200000)
    loan_term = st.sidebar.number_input('Loan Term (years)', min_value=0, value=30)
    employment_years = st.sidebar.number_input('Employment Years', min_value=0, value=5)
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=35)
    code_gender_encoded = st.sidebar.selectbox('Gender', options=[0, 1],
                                               format_func=lambda x: 'Female' if x == 0 else 'Male')
    education_type_encoded = st.sidebar.selectbox('Education Type', options=[0, 1, 2, 3], format_func=lambda x:
    ['Secondary', 'Higher education', 'Incomplete higher', 'Lower secondary'][x])
    own_car_encoded = st.sidebar.selectbox('Owns a Car', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    own_realty_encoded = st.sidebar.selectbox('Owns Real Estate', [0, 1],
                                              format_func=lambda x: 'No' if x == 0 else 'Yes')
    new_ext_source = st.sidebar.slider('External Source Rating', 0.0, 1.0, 0.35)
    payment_rate = st.sidebar.slider('Payment Rate', 0.0, 1.0, 0.03)

    # Prediction
    if st.sidebar.button('Predict Risk'):
        features_pred = {
            'CNT_CHILDREN': cnt_children,
            'Annual_Income': annual_income,
            'Loan_Amount': loan_amount,
            'Loan_Term': loan_term,
            'Employment_Years': employment_years,
            'Age': age,
            'CODE_GENDER_encoded': code_gender_encoded,
            'NAME_EDUCATION_TYPE_encoded': education_type_encoded,
            'FLAG_OWN_CAR_encoded': own_car_encoded,
            'FLAG_OWN_REALTY_encoded': own_realty_encoded,
            'NEW_EXT_SOURCE': new_ext_source,
            'PAYMENT_RATE': payment_rate
        }
        risk_status = predict_single_instance(model, features_pred)
        st.success(f'Risk Status: {risk_status}')
        # Display result with color coding
        if risk_status == 'Low Risk':
            st.markdown(f'<h2 style="color:green;">{risk_status}</h2>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h2 style="color:red;">{risk_status}</h2>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

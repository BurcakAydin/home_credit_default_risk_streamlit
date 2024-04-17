import streamlit as st
import pandas as pd
import lightgbm as lgb


@st.cache(allow_output_mutation=True)
def load_model(model_path):
    # Load the LightGBM model from the specified file
    return lgb.Booster(model_file=model_path)

@st.cache(allow_output_mutation=True)
def validate_data(data):
    # Example validation: ensure no negative values for numerical inputs
    for col in ['CNT_CHILDREN', 'Annual_Income', 'Loan_Amount', 'Loan_Term', 'Employment_Years', 'Age']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if data[col] < 0:
            st.error(f"Error: {col} must be a non-negative number.")
            return None
    return data

@st.cache(allow_output_mutation=True)
def main():
    st.set_page_config(layout="wide")  # Optimize layout for wide screen
    logo_path = "path/to/logo.png"  # Change this path to your actual logo path
    logo = st.sidebar.image(logo_path, width=100)  # Adjust width as necessary

    st.title("Loan Evaluation Dashboard")
    model = load_model('model.txt')  # Load the model at startup

    # Creating a form for user inputs
    with st.form("user_input_form", clear_on_submit=False):
        st.subheader("Please enter the loan application details:")

        # Layout control: organize inputs in pairs using columns
        col1, col2 = st.columns(2)
        with col1:
            cnt_children = st.number_input("Number of Children", min_value=0, step=1)
            annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0, format='%f')
            loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0, format='%f')
            loan_term = st.number_input("Loan Term (in years)", min_value=0, step=1)

        with col2:
            employment_years = st.number_input("Years of Employment", min_value=0, step=1)
            age = st.number_input("Age", min_value=18, step=1)
            code_gender_encoded = st.selectbox("Gender", options=[0, 1],
                                               format_func=lambda x: "Female" if x == 0 else "Male")
            education_type_encoded = st.selectbox("Education Type", [0, 1, 2, 3], format_func=lambda x:
            ["Secondary", "Higher education", "Incomplete higher", "Lower secondary"][x])

        col3, col4 = st.columns(2)
        with col3:
            own_car_encoded = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            new_ext_source = st.slider("External Source Rating", 0.0, 1.0, 0.5)

        with col4:
            own_realty_encoded = st.selectbox("Owns Real Estate", [0, 1],
                                              format_func=lambda x: "No" if x == 0 else "Yes")
            payment_rate = st.slider("Payment Rate", 0.0, 1.0, 0.5)

        submitted = st.form_submit_button("Submit")
        if submitted:
            data = {
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

            validated_data = validate_data(data)
            if validated_data is not None:
                # Convert dictionary to DataFrame
                input_df = pd.DataFrame([validated_data])
                # Predict using the model
                prediction = model.predict(input_df)
                st.success(f"Prediction Result: {prediction[0]}")  # Displaying the prediction result


if __name__ == "__main__":
    main()

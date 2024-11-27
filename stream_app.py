import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load the model and DictVectorizer from a pickle file
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def main():
    # Load images
    image = Image.open('images/icone.png')
    image2 = Image.open('images/image.png')
    
    # Display the images
    st.image(image, use_column_width=False)
    
    # Sidebar options
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch")
    )
    
    # Sidebar info
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    
    # Title
    st.title("Predicting Customer Churn")
    
    # Online Prediction Section
    if add_selectbox == 'Online':
        # Collect input from the user
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen = st.selectbox('Customer is a senior citizen:', [0, 1])
        partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
        phoneservice = st.selectbox('Customer has phone service:', ['yes', 'no'])
        multiplelines = st.selectbox('Customer has multiple lines:', ['yes', 'no', 'no_phone_service'])
        internetservice = st.selectbox('Customer has internet service:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity = st.selectbox('Customer has online security:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox('Customer has online backup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox('Customer has device protection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox('Customer has tech support:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox('Customer has streaming TV:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox('Customer has streaming movies:', ['yes', 'no', 'no_internet_service'])
        contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox('Customer has paperless billing:', ['yes', 'no'])
        paymentmethod = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges
        
        # Prediction output
        output = ""
        output_prob = ""
        
        # Prepare input dictionary
        input_dict = {
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        # Prediction button
        if st.button("Predict"):
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
        
        # Show prediction result
        st.success(f'Churn: {output}, Risk Score: {output_prob}')
    
    # Batch Prediction Section
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload a CSV file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            # Apply transformation and prediction on the data
            X = dv.transform(data.to_dict(orient='records'))
            y_pred = model.predict_proba(X)[:, 1]
            churn = y_pred >= 0.5
            churn = churn.astype(bool)
            data['Churn Prediction'] = churn
            st.write(data)

# Run the app
if __name__ == '__main__':
    main()

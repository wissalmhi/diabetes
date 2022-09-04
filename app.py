import streamlit as st
import pandas as pd
import joblib
import xgboost

st.title('Diabetes prediction')
st.sidebar.markdown("""## please enter the following information:""")

def user_input_features():
    Pregnancies = st.number_input('Insert the number of pregnancies')
    Glucose = st.number_input('Insert the clucose level')
    BloodPressure = st.number_input('Insert the blood pressure')
    SkinThickness = st.number_input('Insert the skin thickness')
    Insulin = st.number_input('Insert the Insulin level')
    BMI = st.number_input('Insert the BMI')
    DiabetesPedigreeFunction = st.number_input('Insert the Diabetes Pedigree Function')
    Age = st.number_input('Insert the Age')
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin':Insulin,
            'BMI':BMI,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
            'Age':Age}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model_VotingEnsemble=joblib.load("model.pkl")
prediction = model_VotingEnsemble.predict(df)
prediction_proba = model_VotingEnsemble.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(model_VotingEnsemble.classes_))

st.subheader('Prediction')
if prediction==1:
    st.caption("the outcome is equal to 1, i.e, this person has diabetes")
else :
    st.caption("the outcome is equal to 0, i.e, this person doesn't have diabetes")





import streamlit as st
import pandas as pd
import joblib 
import numpy as np
import time
 

st.set_page_config(page_title="Car Price AI", page_icon="", layout="wide")


st.markdown("""
<style>

.stApp{
background: linear-gradient(120deg,#1f4037,#99f2c8);
}

/* Header */
.main-title{
text-align:center;
font-size:42px;
font-weight:700;
color:white;
margin-bottom:5px;
}

.sub-title{
text-align:center;
font-size:18px;
color:#f0f0f0;
margin-bottom:40px;
}

/* Sidebar */
[data-testid="stSidebar"]{
background: rgba(255,255,255,0.15);
backdrop-filter: blur(10px);
}

/* Button */
.stButton>button{
width:100%;
height:50px;
font-size:18px;
font-weight:600;
border-radius:10px;
background: linear-gradient(90deg,#ff512f,#dd2476);
color:white;
border:none;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

/* Result Card */
.result-card{
background:white;
padding:40px;
border-radius:20px;
text-align:center;
box-shadow:0px 10px 30px rgba(0,0,0,0.2);
}

.price{
font-size:50px;
font-weight:700;
color:#ff512f;
}

.tag{
color:gray;
font-size:16px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load("car_price_model.pkl")
    poly = joblib.load("car_poly_transform.pkl")
    columns = joblib.load("car_model_columns.pkl")
    return model, poly, columns

model, poly, columns = load_assets()

st.markdown("<div class='main-title'>AI Car Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning Based Car Value Estimator</div>", unsafe_allow_html=True)


st.sidebar.header("🚘 Enter Car Details")

km_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 20000, step=1000)

car_age = st.sidebar.number_input("Car Age (Years)", 0, 30, 3)

brand_list = sorted([c.replace("brand_","") for c in columns if "brand_" in c])
selected_brand = st.sidebar.selectbox("Brand", brand_list)

selected_fuel = st.sidebar.selectbox(
"Fuel Type",
["Petrol","Diesel","CNG","Electric"]
)

selected_seller = st.sidebar.selectbox(
"Seller Type",
["Individual","Dealer","Trustmark Dealer"]
)

selected_owner = st.sidebar.selectbox(
"Owner",
["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"]
)

transmission_type = st.sidebar.radio(
"Transmission",
["Manual","Automatic"]
)

predict = st.sidebar.button("🔮 Predict Car Price")


if predict:

    with st.spinner("Analyzing Market Value..."):
        time.sleep(1)

    input_df = pd.DataFrame(0,index=[0],columns=columns)

    input_df["km_driven"] = km_driven
    input_df["Car_Age"] = car_age

    if f"fuel_{selected_fuel}" in columns:
        input_df[f"fuel_{selected_fuel}"] = 1

    if f"brand_{selected_brand}" in columns:
        input_df[f"brand_{selected_brand}"] = 1

    if f"seller_type_{selected_seller}" in columns:
        input_df[f"seller_type_{selected_seller}"] = 1

    if f"owner_{selected_owner}" in columns:
        input_df[f"owner_{selected_owner}"] = 1

    if transmission_type == "Manual":
        input_df["transmission_Manual"] = 1

    poly_input = poly.transform(input_df)

    prediction = model.predict(poly_input)[0]

    st.markdown(
    f"""
    <div class='result-card'>
        <h3>Estimated Car Price</h3>
        <div class='price'>₹ {max(0,prediction):,.0f}</div>
        <p class='tag'>AI Powered Valuation</p>
    </div>
    """,
    unsafe_allow_html=True
    )



st.markdown(
"""
<br><br>
<center style='color:white;font-size:14px'>
© Made by Hariom Patidar 2026 AI Car Valuation System
</center>
""",
unsafe_allow_html=True
)

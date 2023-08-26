import streamlit as st

st.set_page_config(page_title='Medical Health Buddy', page_icon='Resources/logo.png', layout="wide", initial_sidebar_state="collapsed")

st.markdown('''<style>
.button {
  display: inline-block;
  border-radius: 4px;
  background-color: #f4511e;
  border: none;
  color: #FFFFFF;
  text-align: center;
  font-size: 28px;
  padding: 20px;
  width: 45%;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
}

.button span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button span:after {
  content: '\\00bb';
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.button:hover span {
  padding-right: 25px;
}

.button:hover span:after {
  opacity: 1;
  right: 0;
}

.center {
  text-align: center;
}
</style>''', unsafe_allow_html=True)

st.markdown('''
<div class="center">
    <h1>Medical Health Buddy</h1>
</div>
''', unsafe_allow_html=True)

st.markdown('''
<a href="diabetes" class="button" style="color: white" target="_self"><span>Diabetes Predictor </span></a>
<a href="disease_diagnosis" class="button" style="color: white" target="_self"><span>General Disease Diagnosis </span></a>
<a href="heart_disease" class="button" style="color: white" target="_self"><span>Heart Disease Predictor </span></a>
<a href="liver" class="button" style="color: white" target="_self"><span>Liver Disease Predictor </span></a>
<a href="mental_health" class="button" style="color: white" target="_self"><span>Mental Health Predictor </span></a>  
''', unsafe_allow_html=True)

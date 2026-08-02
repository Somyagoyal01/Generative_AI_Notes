import streamlit as st
st.write("Hello World")
st.title("Hello Streamlit")
st.write("This is my first app")
st.header("Welcome to streamlit")
st.subheader("This is the subheader")
st.text("This is plain text")
#Buttons,checkboxes,sliders
if st.button("Click me!"):
    st.write("Button clicked")
agree=st.checkbox("I agree")
if agree:
    st.write("You agreed")
level=st.slider("Select a Level",1,10,5)
st.write(f"Selected level is {level}")
uploaded_file=st.file_uploader("Upload a file",type=["csv","txt"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df.head())
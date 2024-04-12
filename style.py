import streamlit as st

def add_bg_color():
    st.markdown(
        """
        <style>
        .stApp {
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def custom_radio_horizontal():
    st.markdown(
        """
        <style>
        div.row-widget.stRadio > div {
            flex-direction: row !important;
        }
        label.css-15tx938.e8zbici2 {
            margin-right: 1rem !important;
            background-color: #E1E3E8;
            padding: 5px 10px;
            border-radius: 20px;
        }
        /* Active radio button label */
        div.stRadio > div[role='radiogroup'] > label[data-baseweb='radio'] > div:first-child {
            background-color: #4C78A8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def customize_sidebar():
    st.markdown(
        """
        <style>
        .css-1d391kg {
            background-color: #E1E3E8 !important;
        }
        .css-1aumxhk {
            background-color: #E1E3E8;
            border-color: #E1E3E8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def styles():
    add_bg_color()
    custom_radio_horizontal()
    customize_sidebar()

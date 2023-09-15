# Import streamlit
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
from fastai.vision.all import *
from fastai.learner import load_learner

#######################################################################################################################

def upload_photo(model=None, vocab=None, key=None):
    options = st.multiselect("**Find & select multiple names, then perform Web Search, Download, Upload & Detect...**",
                                 vocab,
                                 vocab[100:108],
                                 key="birds_detector 1")
    st.text(f"Your selection: {options}")

    uploaded_image = st.file_uploader(
        "**Upload an image**", type=["jpg", "png", "jpeg"], key=key)
    st.divider()

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        if st.button("**Detect**", type="primary"):
            output = model.predict(image)
            st.markdown(f"""
                            <div style="text-align:center;">
                            <h1>{output[0]}</h1>
                            </div>""",
                        unsafe_allow_html=True)

            st.image(
                image, caption=f'{output[0]} {max(output[2]).item() * 100:.2f}%', use_column_width=True)

#######################################################################################################################

def capture_photo(model=None, vocab=None, key=None):
    capture_toggle = st.toggle(
    label="**`Enable Camera`**", key="birds_capture_photo")

    if capture_toggle:
        # Check if the cancel checkbox is not selected
        img_file_buffer = st.camera_input(
            label="Take a picture (`try to keep the subject at the center`)", key=key)

        if img_file_buffer:
            # To read image file buffer as a PIL Image:
            image = Image.open(img_file_buffer)

            st.image(image, use_column_width=True)

            if st.button(label="Detect", key="pets_capture_detect"):

                output = model.predict(image)
                st.markdown(f"""<div style="text-align:center;">
                                <h1>{output[0]}</h1>
                                </div>""",
                            unsafe_allow_html=True)

                st.image(
                    image, caption=f'{output[0]} {max(output[2]).item() * 100:.2f}%', use_column_width=True)
    
#######################################################################################################################

def model_info():
    # Model performance on Freezed Layers
    st.subheader("Model performance with Resnet50 (freezed layers)")
    freezed_data = {
        'epoch': [0, 1, 2, 3, 4],
        'train_loss': [1.280186, 0.786523, 0.491951, 0.343532, 0.297919],
        'valid_loss': [0.447442, 0.183758, 0.107396, 0.067139, 0.058960],
        'accuracy': [0.870476, 0.948190, 0.969524, 0.984381, 0.985905],
        'error_rate': [0.129524, 0.051810, 0.030476, 0.015619, 0.014095],
        'time': ['13:50', '11:06', '11:17', '10:44', '11:12']
    }

    df = pd.DataFrame(freezed_data)
    st.table(df)

    # Model performance on Unfreezed Layers
    st.subheader("Model performance with Resnet50 (unfreezed layers)")
    # Create a DataFrame with the provided data
    unfreezed_data = {
        'epoch': [0, 1, 2, 3, 4],
        'train_loss': [0.989004, 0.699652, 0.447756, 0.258508, 0.160852],
        'valid_loss': [0.391414, 0.176702, 0.089814, 0.037954, 0.029704],
        'accuracy': [0.896381, 0.949714, 0.975238, 0.991238, 0.992762],
        'error_rate': [0.103619, 0.050286, 0.024762, 0.008762, 0.007238],
        'time': ['17:05', '12:41', '12:12', '14:43', '12:30']
    }

    df = pd.DataFrame(unfreezed_data)

    # Create a Streamlit table
    st.table(df)

#######################################################################################################################
# Import streamlit
import streamlit as st
from streamlit_option_menu import option_menu
from fastai.vision.all import *
from fastai.learner import load_learner
from bird_species import model_info
from bird_species import capture_photo
from bird_species import upload_photo

def app():
    #######################################################################################################################

    # Set the page config
    st.set_page_config(
        page_title="Bird 525 Species Detector",  # The title of the web page
        page_icon="🕊️",  # The icon of the web page, can be an emoji or a file path
        initial_sidebar_state="collapsed"
    )

    #######################################################################################################################

    st.markdown("<h1 style='text-align: center;'>🕊️Birds 525 Species Detector🕊️</h1>", unsafe_allow_html=True)

    #######################################################################################################################
    
    # Options Menu at the top of the homepage
    selected = option_menu(None, ["Upload", "Capture", "Model"],
                        icons=["cloud upload", "camera", "gear"],
                        menu_icon="cast", default_index=0, orientation="horizontal")

    #######################################################################################################################

    # Load model and model class labels (vocab)
    model = load_learner(fname="models/birds_learner.pkl")

    with open("models/birds_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    # Sorting
    vocab = sorted(vocab)

    #######################################################################################################################

    if selected == "Upload":
        st.caption("""Our project utilizes FastAI Vision with the ResNet50 architecture to classify 
                   525 bird species. Our dataset comprises 84,635 training images, 2,625 test images and 2,625 validation 
                   images, all standardized to 224x224x3 pixels. Initial training yields 96.6% accuracy, improved to 98% post 
                   fine-tuning. Despite gender imbalances, it's a valuable resource for accurate bird species classification.""")       
        
        upload_photo(model = model, vocab=vocab, key="upload photo")

    #######################################################################################################################

    if selected == "Capture":
        capture_photo(model=model, vocab=None, key="capture photo")

    if selected == "Model":
        model_info()

#######################################################################################################################
if __name__ == "__main__":
    app()

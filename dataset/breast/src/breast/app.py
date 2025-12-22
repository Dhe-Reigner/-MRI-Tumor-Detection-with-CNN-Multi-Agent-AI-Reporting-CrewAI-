import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from PIL import Image
from main import run as run_crew

from rag.vectorstore import build_vectorstore
from rag.retriever import retrieve_context
from rag.generator import generate_medical_guidance


#from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Load Modal
model = load_model('bestmodel.keras')

def main(vectorstore):
    st.set_page_config(page_title='MRI Brain Tumor Detection',page_icon='🧠')
    st.title('🧠MRI Brain Tumor Detection')


    # File Uploader
    with st.sidebar:
        st.write('Upload an MRI image to detect whether a tumor is present')

        uploaded_file = st.file_uploader(
            'Upload MRI Image',
            type=['jpg','jpeg','png']
        )

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI', use_container_width=True)

        # preprocessing Image
        img = image.resize((224,224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)[0][0]

        #st.write("### Preiction Score: '{pred:.4f}'")
        st.metric('Tumor Probability', f'{pred*100:.2f}%')

        # Result
        if  pred > 0.5:
            st.error("🧠 Tumor Detected (Malignant)")
        else:
            st.success("✅ No Tumor Detected (Benign)")

        with st.spinner('AI Radiology team generating report...'):
            report  = run_crew(pred)
        
        label = 'malignant' if pred > 0.5 else 'benign'

        query = f"""
        MRI brain tumor detected as {label}.
        Provide educational guidance and general information.
        """

        with st.spinner('Generating Educational Medical Guidance...'):
            context = retrieve_context(vectorstore, query)
            rag_response = generate_medical_guidance(pred,context)

        st.subheader('🩺 AI- Generated Breast MRI Report')
        #st.write(report.raw)
        st.write(rag_response)

        st.caption(
            '⚠️ This system is for research and educational purposes only.'
            'It is not a substitute for professional medical diagnosis.'
            )
        
if __name__=='__main__':
    main()
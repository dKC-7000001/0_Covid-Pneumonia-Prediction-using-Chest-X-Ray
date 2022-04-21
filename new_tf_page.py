import streamlit as st
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./updated_tf_model.h5')
    return model

with st.spinner('Model is being loaded...'):
    model = load_model()

st.title("Covid Detection using Chest X-ray // by_dKC")
st.write("""### Upload image to detect disease""")

image_data = st.file_uploader('Please upload chest X-ray image')

def import_and_predict(image_data, model):

    image1 = Image.open(image_data)
    scaled_image = tf.image.resize(image1, 
    size = (224,224), method='bilinear', preserve_aspect_ratio=False,
    antialias=False, name=None)

    f = np.reshape(scaled_image, (1,224,224,3))
    #ans1 = f.shape
    #st.subheader(ans1)
    #st.subheader(f)
    #ans2 = model.predict(f)
    #z = np.random.rand(1,224,224,3)
    return model.predict(f)

if image_data is None:
    st.text('Please uplaod x-ray file')

else:
    #st.write('This codeblock is running')
    image2 = Image.open(image_data)
    st.image(image_data, use_column_width=True)
    prediction = import_and_predict(image_data, model)
    ans3 = np.argmax(prediction)
    #st.write(prediction)
    #st.write(ans3)

    if ans3 == 0:
        st.subheader('Possible Chance of Covid')
    elif ans3 == 1:
        st.subheader('X-ray looks Normal, enjoy :)')
    elif ans3 == 2:
        st.subheader('Possible Chance of Viral Pneumonia')
    else:
        st.subheader('Dont know :((')

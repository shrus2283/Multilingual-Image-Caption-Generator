import streamlit as st
from googletrans import Translator
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pickle
from PIL import Image

# Load the VGG16 model
vgg_model = VGG16()
# Restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load the preprocessed image features
with open('working/features.pkl', 'rb') as f:
    features = pickle.load(f)

# Load the caption mapping
with open('flickr8k/captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

# Create mapping of image to captions
mapping = {}
# Process lines
for line in captions_doc.split('\n'):
    # Split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # Remove extension from image ID
    image_id = image_id.split('.')[0]
    # Convert caption list to string
    caption = " ".join(caption)
    # Create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # Store the caption
    mapping[image_id].append(caption)

# Clean the captions
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # Take one caption at a time
            caption = captions[i]
            # Preprocessing steps
            # Convert to lowercase
            caption = caption.lower()
            # Delete digits, special chars, etc.
            caption = caption.replace('[^A-Za-z]', '')
            # Delete additional spaces
            caption = caption.replace('\s+', ' ')
            # Add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

clean(mapping)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([caption for captions in mapping.values() for caption in captions])
vocab_size = len(tokenizer.word_index) + 1

# Get maximum length of the caption available
max_length = max(len(caption.split()) for captions in mapping.values() for caption in captions)

# Load the trained model
myModel = tf.keras.models.load_model('working/accurate_model.h5')

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption
def predict_caption(myModel, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'
    # Iterate over the max length of sequence
    for _ in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict next word
        yhat = myModel.predict([image, sequence], verbose=0)
        # Get index with high probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        # Append word as input for generating next word
        in_text += " " + word
        # Stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


def preprocess_image(image):
    # Resize the image to a fixed size
    image = image.resize((224, 224))
    # Convert image to numpy array
    image = np.array(image)
    # Expand dimensions to match the batch size expected by the model
    image = np.expand_dims(image, axis=0)
    # Preprocess the image (normalize pixel values)
    image = preprocess_input(image)
    return image


# Translate the caption
def translate_caption(caption, target_language):
    translator = Translator()
    translation = translator.translate(caption, dest=target_language)
    return translation.text


# Streamlit app
def main():
    st.title("Image Captioning")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Language selection
    target_language = st.selectbox("Select Target Language", ["English", "Spanish", "French", "German", "Japanese"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption button
        if st.button("Generate Caption"):
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Extract features using VGG16
            features = vgg_model.predict(processed_image, verbose=0)

            # Generate caption
            caption = predict_caption(myModel, features, tokenizer, max_length)

            # Translate caption
            if target_language != "English":
                caption = translate_caption(caption, target_language.lower())

            # Display the predicted caption
            st.success("Caption: " + caption)

if __name__ == '__main__':
    main()

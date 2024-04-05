import streamlit as st
import onnxruntime
from PIL import Image
import io
import numpy as np

# Load the model
model = onnxruntime.InferenceSession(r'runs\classify\train\weights\best.onnx')

# Get the input names
input_names = [input.name for input in model.get_inputs()]

# Print the input names (for debugging)
#print("Model Input Names:", input_names)

# Define a function for prediction
def predict(uploaded_file):
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)  # Open the uploaded image

            # Resize the image to match the expected dimensions (e.g., 3x224x224)
            target_size = (224, 224)  # Adjust based on model requirements
            image = image.resize(target_size)

            # Convert image to numpy array and normalize pixel values to [0, 1]
            image_array = np.array(image) / 255.0

            # Ensure correct channel order (RGB to BGR if needed)
            if model.get_inputs()[0].shape[1] == 3:  # Check if model expects BGR
                image_array = np.transpose(image_array, (2, 0, 1))  # Convert RGB to BGR

            # Add batch dimension at the beginning (index 0)
            image_array = np.expand_dims(image_array, axis=0)

            # Run prediction
            predictions = model.run(None, {"images": image_array.astype(np.float32)})

            # Handle model outputs (display, save, etc.)
            return predictions

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    else:
        st.warning("Please upload an image to make a prediction.")
        return None

st.set_page_config(
    page_title="Med_Leaf Identifier",
    page_icon="🌿"
)

st.title('Med_Leaf Identifier 🌿')

st.markdown('This is an application for Medicinal leaf identification using YOLO')

# File uploader for image
uploaded_files = st.file_uploader(label="Choose image files",
                                   type=['png', 'jpg', 'jpeg'],
                                   accept_multiple_files=True)

# Dictionary representing leaf mapping
leaf_mapping = {
    0: "Alpinia Galanga (Rasna)",
    1: "Amaranthus Viridis (Arive-Dantu)",
    2:"Artocarpus Heterophyllus (Jackfruit)",
    3:"Azadirachta Indica (Neem)",
    4:"Basella Alba (Basale)",
    5:"Brassica Juncea (Indian Mustard)",
    6:"Carissa Carandas (Karanda)",
    7:"Citrus Limon (Lemon)",
    8:"Ficus Auriculata (Roxburgh fig)",
    9:"Ficus Religiosa (Peepal Tree)",
    10:"Hibiscus Rosa-sinensis",
    11:"Jasminum (Jasmine)",
    12:"Mangifera Indica (Mango)",
    13:"Mentha (Mint)",
    14:"Moringa Oleifera (Drumstick)",
    15:"Muntingia Calabura (Jamaica Cherry-Gasagase)",
    16:"Murraya Koenigii (Curry)",
    17:"Nerium Oleander (Oleander)",
    18:"Nyctanthes Arbor-tristis (Parijata)",
    19:"Ocimum Tenuiflorum (Tulsi)",
    20:"Piper Betle (Betel)",
    21:"Plectranthus Amboinicus (Mexican Mint)",
    22:"Pongamia Pinnata (Indian Beech)",
    23:"Psidium Guajava (Guava)",
    24:"Punica Granatum (Pomegranate)",
    25:"Santalum Album (Sandalwood)",
    26:"Syzygium Cumini (Jamun)",
    27:"Syzygium Jambos (Rose Apple)",
    28:"Tabernaemontana Divaricata (Crape Jasmine)",
    29:"Trigonella Foenum-graecum (Fenugreek)"

    # Add more mappings as needed
    
}

# Function to get leaf name by index
def get_leaf_name(index):
    return leaf_mapping.get(index, f"Unknown Leaf {index}")

# Example usage
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)  # Load image using PIL
        st.image(image, caption="Uploaded Image")

        # Run prediction
        predictions = predict(uploaded_file)

        # Display results (implementation-specific)
        if predictions is not None:
            # Find the index with the highest value in the prediction array
            max_index = np.argmax(predictions[0][0])

            # Get the corresponding leaf name
            predicted_leaf_name = get_leaf_name(max_index)

            # Print the predicted leaf name
            st.write(f"Predicted Leaf Name: {predicted_leaf_name}")
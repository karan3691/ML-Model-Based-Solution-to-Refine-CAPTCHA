import streamlit as st
import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from captcha.image import ImageCaptcha
from io import BytesIO
from PIL import Image
import cv2
import os
from adversarial import create_adversarial_pattern

# Load the trained model
try:
    model = load_model('captcha_solver_model.h5')
    model_loaded = True
except:
    st.warning("Model not found. Will generate CAPTCHA without ML-based enhancements.")
    model_loaded = False

# Character set must match the one used during training
char_set = list(string.ascii_uppercase + string.digits)

def generate_captcha_text(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def predict_captcha(image):
    """
    Use the trained model to predict CAPTCHA text
    """
    if not model_loaded:
        return None
        
    # Convert PIL image to numpy array, ensure it's RGB, and normalize
    # Ensure exact same preprocessing as training
    image = image.convert('RGB')
    image = image.resize((200, 100))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict character probabilities
    predictions = model.predict(img_array, verbose=0)
    
    # Convert predictions to characters with confidence scores
    predicted_text = ""
    confidence_scores = []
    
    for pred in predictions:
        char_probs = pred[0]
        char_idx = np.argmax(char_probs)
        confidence = float(char_probs[char_idx])
        predicted_text += char_set[char_idx]
        confidence_scores.append(confidence)
        
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    return predicted_text, avg_confidence

def apply_adversarial_noise(image, true_text, epsilon=0.05):
    """
    Apply adversarial noise to the CAPTCHA image to make it harder for bots to solve
    """
    if not model_loaded:
        return image
        
    # Convert PIL image to tensor, ensure it's RGB
    image = image.convert('RGB')
    image = image.resize((200, 100))
    img_array = img_to_array(image) / 255.0
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))
    
    # Create one-hot encoded true labels
    true_labels = []
    for char in true_text:
        try:
            index = char_set.index(char)
            one_hot = tf.zeros((1, len(char_set)))
            one_hot = tf.tensor_scatter_nd_update(one_hot, [[0, index]], [1.0])
            true_labels.append(one_hot)
        except ValueError:
            # Handle case where character is not in char_set
            one_hot = tf.zeros((1, len(char_set)))
            true_labels.append(one_hot)
    
    try:
        # Generate adversarial image
        adversarial_img_tensor = create_adversarial_pattern(model, img_tensor, true_labels, epsilon)
        
        # Convert tensor back to PIL image
        adversarial_img_array = adversarial_img_tensor.numpy()[0] * 255.0
        adversarial_img_array = np.clip(adversarial_img_array, 0, 255).astype('uint8')
        adversarial_image = Image.fromarray(adversarial_img_array)
        return adversarial_image
    except Exception as e:
        # If adversarial generation fails, return original image
        print(f"Error in adversarial generation: {e}")
        return image

def generate_captcha_image(text, difficulty=1, use_adversarial=True):
    """
    Generate a CAPTCHA image with ML-based enhancements
    """
    # Create basic CAPTCHA image
    image_captcha = ImageCaptcha(width=200, height=100)
    image = image_captcha.generate_image(text)
    
    # Convert to RGB to ensure consistency
    image = image.convert('RGB')
    
    # Apply adversarial perturbations based on difficulty
    if model_loaded and use_adversarial and difficulty > 1:
        # Scale epsilon with difficulty - lower values for better readability
        epsilon = 0.005 * difficulty  
        image = apply_adversarial_noise(image, text, epsilon)
    
    # Add visual noise based on difficulty
    if difficulty > 1:
        # Convert back to RGBA for drawing
        image = image.convert("RGBA")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
        height, width, _ = image_cv.shape
        
        # Add random lines based on difficulty
        for _ in range(difficulty * 3):  # Reduced from 5 to 3 for better readability
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            # Use semi-transparent colors for less interference
            alpha = random.randint(100, 200)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), alpha)
            thickness = 1 if difficulty < 4 else 2
            cv2.line(image_cv, (x1, y1), (x2, y2), color, thickness)
        
        image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGBA))
    
    # Final conversion to RGB for consistent format
    display_image = image.convert('RGB')
    
    return display_image

def main():
    st.title("ML-Enhanced CAPTCHA System")
    st.write("This system uses machine learning and adversarial techniques to challenge bots.")
    
    # Sidebar controls
    st.sidebar.title("Settings")
    difficulty = st.sidebar.slider("CAPTCHA Difficulty", 1, 5, 1)
    use_ml = st.sidebar.checkbox("Use ML Model", value=True)
    use_adversarial = st.sidebar.checkbox("Use Adversarial Examples", value=True)
    
    # Information about ML integration
    if model_loaded:
        st.sidebar.success("ML Model loaded successfully")
    else:
        st.sidebar.error("ML Model not available")
        use_ml = False
        use_adversarial = False
    
    # Generate new CAPTCHA text if needed
    if 'captcha_text' not in st.session_state:
        st.session_state.captcha_text = generate_captcha_text()
    
    captcha_text = st.session_state.captcha_text
    captcha_image = generate_captcha_image(captcha_text, difficulty, use_adversarial)
    
    # Display CAPTCHA image
    buf = BytesIO()
    captcha_image.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="CAPTCHA Challenge")
    
    # Show ML model prediction (for demonstration purposes)
    if use_ml and model_loaded and st.sidebar.checkbox("Show Model Prediction (Demo)", value=True):
        try:
            predicted_text, avg_confidence = predict_captcha(captcha_image)
            if predicted_text:
                st.sidebar.info(f"Model prediction: {predicted_text}")
                st.sidebar.write(f"Actual text: {captcha_text}")
                
                # Calculate character-level accuracy
                char_matches = sum([a==b for a,b in zip(predicted_text, captcha_text)])
                accuracy = char_matches / len(captcha_text)
                st.sidebar.write(f"Character accuracy: {char_matches}/{len(captcha_text)} ({accuracy:.0%})")
                
                # Display confidence score
                st.sidebar.write(f"Confidence score: {avg_confidence:.2f}")
                st.sidebar.progress(avg_confidence)
            else:
                st.sidebar.warning("Could not generate prediction")
        except Exception as e:
            st.sidebar.error(f"Error in prediction: {str(e)}")
    
    # User input
    user_input = st.text_input("Enter the text you see above:")
    
    if st.button("Submit"):
        # Strip whitespace and normalize case for comparison
        user_text = user_input.strip().upper()
        captcha_text_clean = captcha_text.strip()
        
        if user_text == captcha_text_clean:
            st.success("CAPTCHA validated successfully!")
            st.session_state.captcha_text = generate_captcha_text()  
        else:
            st.error(f"Incorrect CAPTCHA. Please try again.")
            # For debugging
            if st.sidebar.checkbox("Debug Mode", value=False):
                st.sidebar.write(f"User input: '{user_text}'")
                st.sidebar.write(f"Expected: '{captcha_text_clean}'")
            st.session_state.captcha_text = generate_captcha_text()

if __name__ == "__main__":
    main()

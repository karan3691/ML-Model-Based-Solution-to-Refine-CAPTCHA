import streamlit as st
import random
import string
from captcha.image import ImageCaptcha
from io import BytesIO
from PIL import Image

# Optionally, you can import adversarial functions and the trained model if needed:
# from tensorflow.keras.models import load_model
# from adversarial import create_adversarial_pattern
# model = load_model('captcha_solver_model.h5')

def generate_captcha_text(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def generate_captcha_image(text, difficulty=1):
    """
    Generate a CAPTCHA image.
    
    Args:
        text (str): The CAPTCHA text.
        difficulty (int): Difficulty level (1-5) which can control added noise/distortions.
        
    Returns:
        image (PIL.Image): The generated CAPTCHA image.
    """
    image_captcha = ImageCaptcha(width=200, height=100)
    image = image_captcha.generate_image(text)
    
    if difficulty > 1:
        image = image.convert("RGBA")
        import numpy as np
        import cv2
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
        height, width, _ = image_cv.shape
        for _ in range(difficulty * 5):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
            cv2.line(image_cv, (x1, y1), (x2, y2), color, 1)
        image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGRA2RGBA))
    
    return image

def main():
    st.title("Enhanced CAPTCHA System")
    st.write("This system uses ML and adversarial techniques to challenge bots.")
    
    difficulty = st.sidebar.slider("CAPTCHA Difficulty", 1, 5, 1)
    
    if 'captcha_text' not in st.session_state:
        st.session_state.captcha_text = generate_captcha_text()
    
    captcha_text = st.session_state.captcha_text
    captcha_image = generate_captcha_image(captcha_text, difficulty)
    
    buf = BytesIO()
    captcha_image.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="CAPTCHA Challenge")
    
    user_input = st.text_input("Enter the text you see above:")
    
    if st.button("Submit"):
        if user_input.upper() == captcha_text:
            st.success("CAPTCHA validated successfully!")
            st.session_state.captcha_text = generate_captcha_text()  
        else:
            st.error("Incorrect CAPTCHA. Please try again.")
            st.session_state.captcha_text = generate_captcha_text()  

if __name__ == "__main__":
    main()

import os
import random
import string
from captcha.image import ImageCaptcha

def random_text(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def generate_captcha(text, save_path):
    image = ImageCaptcha(width=200, height=100)
    image.write(text, save_path)

def main():
    os.makedirs("captcha_dataset", exist_ok=True)
    num_images = 10000 
    for i in range(num_images):
        text = random_text(5)
        filename = f"captcha_dataset/{text}_{i}.png"
        generate_captcha(text, filename)
        if i % 1000 == 0:
            print(f"Generated {i} images...")

if __name__ == "__main__":
    main()

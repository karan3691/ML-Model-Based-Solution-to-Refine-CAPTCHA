# ML Model-Based Solution to Refine CAPTCHA

## Problem Description
This project focuses on enhancing CAPTCHA systems using a machine learning (ML) approach. The goal is to improve security and user experience by:

- **Dataset:** Creating or using a dataset of CAPTCHA images.
- **Model Training:** Training an ML model to generate and validate CAPTCHA challenges.
- **Security Features:** Implementing adversarial techniques to counter automated CAPTCHA-solving bots.
- **Deployment:** Deploying a web-based CAPTCHA generator using Streamlit.

## Project Overview
This project includes:
- **Dataset Generation:** Using the `captcha` library to generate synthetic CAPTCHA images.
- **Data Preparation:** Loading and preprocessing images and encoding labels.
- **Model Training:** Building and training a multi-output Convolutional Neural Network (CNN) to recognize CAPTCHA text.
- **Adversarial Techniques:** Incorporating methods (e.g., FGSM) to generate adversarial examples that challenge automated solvers.
- **Deployment:** A Streamlit app that generates CAPTCHA challenges, validates user input, and allows customization of CAPTCHA difficulty.

## Project Structure

captcha_project/

├── captcha_dataset/ # Folder for generated CAPTCHA images 

├── adversarial.py # Module for generating adversarial examples 

├── app.py # Streamlit app for CAPTCHA generation and validation 

├── data_preparation.py # Module to load and encode CAPTCHA images 

├── generate_dataset.py # Script to generate the CAPTCHA dataset 

├── requirements.txt # List of project dependencies 

└── train_model.py # Script to build and train the CAPTCHA solver model


## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/karan3691/ML-Model-Based-Solution-to-Refine-CAPTCHA.git
   ```
   ```bash
   cd ML-Model-Based-Solution-to-Refine-CAPTCHA
   ```

2. **Installation:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Generate the CAPTCHA Dataset**
   
   **Run the script to create a dataset of CAPTCHA images:**
   ```bash
   python generate_dataset.py
   ```

   This will generate CAPTCHA images and save them in the ```captcha_dataset ```folder.

2. **Train the CAPTCHA Solver Model**

   **Train the ML model to recognize CAPTCHA images by running:**
   ```bash
   python train_model.py
   ```

   The model is saved as ```captcha_solver_model.h5``` after training.

3. **(Optional) Adversarial Techniques**

   **The file ```adversarial.py``` contains a function to generate adversarial examples using FGSM. You can integrate this function into your CAPTCHA generation process to add extra noise that challenges bots.**

4. **Run the Streamlit Application**

   **Start the web application to generate and validate CAPTCHA challenges:**
   ```bash
   streamlit run app.py
   ```

   The app includes:
   - **input Section:** Generates and displays CAPTCHA challenges.
   - **Output Section:** Validates user responses.
   - **Enhancements:** A slider to adjust CAPTCHA difficulty.



## Model Hosting
Due to GitHub's file size restrictions, the trained model ```(captcha_solver_model.h5)``` is hosted on Hugging Face. To download the model within your code, use:
```bash
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="karan3691/captcha-model", filename="captcha_solver_model.h5")
```

Make sure your application references the model from the correct path.

## Acknowledgements
- **Captcha** - Python library for generating CAPTCHA images.
- **Streamlit** - Framework for building interactive web apps
- **Hugging Face** - Platform for hosting and sharing ML models.







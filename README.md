# 🌞 Solar Vision Inference (Low Res) - Streamlit App

A Streamlit-based application for low-resolution solar panel detection using machine learning models. This app is designed to demonstrate inference from a pre-trained model, capable of detecting solar panels in low-resolution images and calculating power generation.

---

### 🌐 Website 
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://solar-vision-inference-low-res.streamlit.app/)

Visit the live demo: [Solar Vision Inference App](https://solar-vision-inference-low-res.streamlit.app/)

---

## 📋 Features

- **Low-resolution image inference**: Detect solar panels from low-res spatial data.
- **Streamlit Interface**: User-friendly interface for uploading and analyzing images.
- **Model Integration**: Utilizes YOLO models via `ultralytics` for efficient inference.

## 🚀 Quickstart

To run the app locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/ViratSrivastava/Solar-Vision-Inference-Low-Res-StreamLit.git
   cd Solar-Vision-Inference-Low-Res-StreamLit
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

The app will open in your default web browser.


## 📦 Requirements

- Python 3.10
- Libraries listed in `requirements.txt`, including:
  - `streamlit`
  - `opencv-python-headless`
  - `ultralytics`
  - `Pillow`
  - `numpy`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

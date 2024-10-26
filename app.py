import streamlit as st
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from PIL import Image
import io
import pandas as pd
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore

# This must be at the very top of your script, outside of any function
st.set_page_config(
    page_title="Solar Vision",
    page_icon="satellite-icon-logo-design-illustration-vector.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

def set_page_container_style():
    st.markdown("""
        <div id="starfield-container">
            <canvas id="starfield"></canvas>
        </div>
        <style>
            .reportview-container {
                background: transparent;
            }
            .main .block-container {
                background: transparent;
            }
        </style>
    """, unsafe_allow_html=True)

def add_starfield_animation():
    with open('starfield.js', 'r') as file:
        js_code = file.read()
    
    st.markdown(f"""
    <script>
    {js_code}
    </script>
    """, unsafe_allow_html=True)

# Load custom CSS
def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def nav_button(title, icon, is_active=False):
    active_class = "active" if is_active else ""
    return f"""
        <button class="nav-button {active_class}">
            <i class="fas {icon}"></i>
            {title}
        </button>
    """

def read_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 9:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            boxes.append((class_id, coords))
    
    return boxes

def normalize_to_pixel_coordinates(box, image_width, image_height):
    return [
        (int(box[i] * image_width), int(box[i+1] * image_height))
        for i in range(0, len(box), 2)
    ]

def mask_outside_boxes(image, labels_file):
    boxes = read_labels(labels_file)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = image.shape[:2]

    for _, box in boxes:
        pixel_coords = normalize_to_pixel_coordinates(box, width, height)
        cv2.fillPoly(mask, [np.array(pixel_coords)], (255, 255, 255))

    mask_inv = cv2.bitwise_not(mask)
    black_bg = np.zeros_like(image)
    result = cv2.bitwise_and(image, image, mask=mask)
    result += cv2.bitwise_and(black_bg, black_bg, mask=mask_inv)

    return result, mask

def calculate_solar_power(mask, total_area_meters):
    total_pixels = mask.size
    black_pixels = np.sum(mask == 0)
    black_percentage = (black_pixels / total_pixels) * 100

    black_area = (black_percentage / 100) * total_area_meters
    colored_area = total_area_meters - black_area

    panel_area = 1.71
    if total_area_meters < panel_area:
        raise ValueError(f"Image area ({total_area_meters:.2f} m²) is too small. Minimum required area is 1.71 m².")

    num_panels = colored_area / panel_area
    power_generated = num_panels * 0.30

    return black_percentage, colored_area, power_generated, num_panels

def footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
    }

    </style>
    <div class="footer">
        <p>Developed by: 
            <a href="https://github.com/ViratSriavstava" target="_blank">Virat Srivastava GitHub</a> | 
            <a href="https://linkedin.com/in/virat-srivastava" target="_blank">Virat Srivastava LinkedIn</a> | 
            <a href="https://github.com/durgesh2411" target="_blank">Durgesh Kumar Singh GitHub</a> | 
            <a href="https://www.linkedin.com/in/durgesh-singh-745263252/" target="_blank">Durgesh Kumar Singh LinkedIn</a> | 
            <a href="https://github.com/VishuKalier2003" target="_blank">Vishu Kalier GitHub</a> | 
            <a href="https://www.linkedin.com/in/durgesh-singh-745263252/" target="_blank">Vishu Kalier LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def process_image(uploaded_file, image_area_x, image_area_y, model):
    if not os.path.exists('input'):
        os.makedirs('input')
    
    input_path = 'input/inference.png'
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    predict_dir = 'runs/obb/predict'
    if os.path.exists(predict_dir):
        shutil.rmtree(predict_dir)

    results = model.predict(
        source=input_path,
        save=True,
        save_txt=True,
        save_json=True,
        show_labels=True
    )

    image = cv2.imread(input_path)
    labels_path = 'runs/obb/predict/labels/inference.txt'
    result, mask = mask_outside_boxes(image, labels_path)

    total_area_meters = image_area_x * image_area_y
    black_percentage, colored_area, power_generated, num_panels = calculate_solar_power(mask, total_area_meters)

    # Read the YOLO output image
    yolo_output_path = 'runs/obb/predict/inference.jpg'
    yolo_output = cv2.imread(yolo_output_path)
    yolo_output_rgb = cv2.cvtColor(yolo_output, cv2.COLOR_BGR2RGB)
    yolo_output_pil = Image.fromarray(yolo_output_rgb)

    # Convert masked result to PIL
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    return result_pil, yolo_output_pil, black_percentage, colored_area, power_generated, num_panels

def model_matrix_page():
    st.title("Madel Matrix and Training Results")

    # Read the CSV file
    try:
        df = pd.read_csv("results.csv")
        
        # Create subplots
        fig = make_subplots(rows=3, cols=2, 
                            subplot_titles=("Training Losses", "Validation Losses", 
                                            "Precision and Recall", "mAP Metrics",
                                            "Learning Rates"),
                            vertical_spacing=0.1)

        # Training losses
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/box_loss'], name='Box Loss', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/cls_loss'], name='Class Loss', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/dfl_loss'], name='DFL Loss', mode='lines'), row=1, col=1)

        # Validation losses
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/box_loss'], name='Val Box Loss', mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/cls_loss'], name='Val Class Loss', mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/dfl_loss'], name='Val DFL Loss', mode='lines'), row=1, col=2)

        # Precision and Recall
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/precision(B)'], name='Precision', mode='lines'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/recall(B)'], name='Recall', mode='lines'), row=2, col=1)

        # mAP Metrics
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/mAP50(B)'], name='mAP50', mode='lines'), row=2, col=2)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['metrics/mAP50-95(B)'], name='mAP50-95', mode='lines'), row=2, col=2)

        # Learning Rates
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr/pg0'], name='LR pg0', mode='lines'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr/pg1'], name='LR pg1', mode='lines'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr/pg2'], name='LR pg2', mode='lines'), row=3, col=1)

        # Update layout
        fig.update_layout(height=1200, title_text="Training Metrics")
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Value")

        # Display the plot, extending from edge to edge
        st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    except FileNotFoundError:
        st.error("Error: 'results.csv' file not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
        # Add a subheader for the training transcript
    st.subheader("Training Transcript")

    # Read the training transcript from the file
    try:
        with open("training_transcript.txt", "r") as file:
            transcript = file.read()
    except FileNotFoundError:
        transcript = "Training transcript file not found."

    # Create a scrollable text area with the transcript
    st.text_area("Training Log", value=transcript, height=400, max_chars=None, key="transcript")



def create_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", ax=ax)
    plt.title("Model Matrix Heatmap")
    return fig


def run_inference_page():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("Solar Panel Analysis")
    st.write("Upload an image and enter its dimensions to analyze solar panel potential.")

    # Load YOLO model
    @st.cache_resource
    def load_model():
        return YOLO('weights/best.pt')
    
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    # Area input
    col1, col2 = st.columns(2)
    with col1:
        image_area_x = st.number_input("Width of area (meters)", min_value=0.0, value=2.0, step=0.1)
    with col2:
        image_area_y = st.number_input("Height of area (meters)", min_value=0.0, value=2.0, step=0.1)

    total_area = image_area_x * image_area_y

    if total_area < 1.71:
        st.error("⚠️ Total area must be at least 1.71 m² for a single solar panel")
        return

    if uploaded_file is not None:
        try:
            with st.spinner("Processing image..."):
                result_image, yolo_output, black_percentage, colored_area, power_generated, num_panels = process_image(
                    uploaded_file, image_area_x, image_area_y, model
                )

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Original & Detection", "Masked Result", "Analysis"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption="Original Image")
                with col2:
                    st.image(yolo_output, caption="YOLO Detection")
            
            with tab2:
                st.image(result_image, caption="Masked Result")
            
            with tab3:
                # Display metrics
                st.subheader("Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Non-Panel Area", f"{black_percentage:.1f}%")
                with col2:
                    st.metric("Panel Area", f"{colored_area:.2f} m²")
                with col3:
                    st.metric("Power Generation", f"{power_generated:.2f} kWh")
                '''
                # Additional details
                st.info(f"Number of solar panels that can be installed: {int(num_panels)}")
                can be used in high
                '''                
                # Detailed calculations
                with st.expander("View Detailed Calculations"):
                    st.write(f"""
                    - Total Area: {total_area:.2f} m²
                    - Area suitable for panels: {colored_area:.2f} m²
                    - Single panel area: 1.71 m²
                    - Number of panels: {int(num_panels)}
                    - Power per panel: 0.30 kWh
                    - Total power generation: {power_generated:.2f} kWh
                    """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

def github_page():

    st.markdown("""
    <h1 style='text-align: center;'>GitHub Repository</h1>
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>Project Repository</h3>
        <p>This project is open source and available on GitHub. You can find the source code, contribute to the project, 
        or report issues through our repository.</p>
        <a href="https://github.com/ViratSrivastava/Solor-Vision-Inference" target="_blank" style="
            display: inline-block;
            padding: 10px 20px;
            background-color: #24292e;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 10px;">
            <i class="fab fa-github"></i> Visit Repository
        </a>
    </div>
    
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
        <h3>Features Implemented</h3>
        <ul>
            <li>YOLOV8-based solar panel detection</li>
            <li>Area calculation and power estimation</li>
            <li>Interactive web interface using Streamlit</li>
            <li>Automated report generation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def welcome_page():
    st.title("Welcome to Solar Vision 🌞")

    st.markdown("### Solar Vision: Harnessing Automated Detection and Quantification of Solar Panels in Urban Areas to Promote Renewable Energy Adoption and Achieve Sustainable Development Goal")
    st.write("Solar Vision is an advanced pipeline that uses artificial intelligence and classical algorithms to analyze Spatial images and assess the urban envirnment for solar panel installation on rooftops or open areas with application of assesment on temporal data for analsysis.")

    st.markdown("### Key Features")
    features = [
        "AI-powered analysis of aerial images",
        "Automatic detection of suitable installation areas",
        "Calculation of Estimated power generation",
        "Estimation of installable solar panel count",
        "Generation of detailed reports"
    ]
    for feature in features:
        st.markdown(f"- {feature}")

    st.markdown("### How to Use")
    steps = [
        "This model is suggested to be used on Low Resolution Spartial Imagery.",
        "Upload a spatial image of the urban area. ",
        "Navigate to the \"Run Inference\" page using the sidebar.",
        "Upload an aerial image of the area you want to analyze",
        "Provide the actual dimensions of the area (if known)",
        "Review the detailed analysis and results",
        "Download the generated report for further use"
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")
    
def set_page_container_style():
    # Add the HTML for the starfield container
    st.markdown("""
        <div id="starfield-container">
            <canvas id="starfield"></canvas>
        </div>
    """, unsafe_allow_html=True)

    # You can also add other styling here if needed
    st.markdown("""
        <style>
            .reportview-container {
                background: transparent;
            }
            .main .block-container {
                background: transparent;
            }
        </style>
    """, unsafe_allow_html=True)

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Welcome"
    
    set_page_container_style()
    add_starfield_animation()
    # st.title("Solar Vision")
    
    # Load custom CSS
    load_css()

    # Sidebar navigation
    st.sidebar.markdown('<h1>Navigation</h1>', unsafe_allow_html=True)
    
    # Navigation options
    pages = {
        "Welcome": welcome_page,
        "Run Inference": run_inference_page,
        "Model Matrix": model_matrix_page,
        "GitHub Repository": github_page
    }

    # Create navigation buttons
    for page, func in pages.items():
        if st.sidebar.button(page, key=page):
            st.session_state.page = page

    # Display the selected page
    if st.session_state.page in pages:
        pages[st.session_state.page]()

    # Load the image
    logo = Image.open("satellite-icon-logo-design-illustration-vector.png")

    # Create three columns in the sidebar
    col1, col2, col3 = st.sidebar.columns([1,2,1])

    # Display the image in the middle column
    with col2:
        st.image(logo, width=150)

    # Add sidebar footer
    st.sidebar.markdown("""
        <div style="margin-top: 20px; text-align: center;">
            <hr>
            <h3>Made with ❤️ by</h3>
            <h5>
            Virat Srivastava<br>
            Durgesh Kumar Singh<br>
            Vishu Kalier
            </h5>
            Version 1.0.0
        </div>
    """, unsafe_allow_html=True)

    # Add interactive background
    st.markdown("""
        <div id="starfield-container">
            <canvas id="starfield"></canvas>
        </div>
    """, unsafe_allow_html=True)
    

    # Add footer
    footer()
if __name__ == "__main__":
    main()
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from PIL import Image
import os
import numpy as np
from flask import Flask, request, render_template
import cv2
from cv2 import cvtColor, Laplacian, COLOR_BGR2GRAY,Canny
import qrcode
from io import BytesIO
from fontTools.ttLib import TTFont
import qrcode
app = Flask(__name__)

# Define the folder for uploading posters
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the "uploads" directory exists, or create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to calculate average RGB values
def calculate_average_rgb(image_path):
    with Image.open(image_path) as img:
        r, g, b = img.convert("RGB").split()
        r_avg = sum(r.getdata()) // len(r.getdata())
        g_avg = sum(g.getdata()) // len(g.getdata())
        b_avg = sum(b.getdata()) // len(b.getdata())
    return r_avg, g_avg, b_avg

def evaluate_indentation(image_path):
    # In this example, we'll use a simple threshold-based evaluation
    img = Image.open(image_path)
    img_data = np.array(img)
    threshold = 150  # Adjust this threshold as needed
    indentation_score = (np.mean(img_data) > threshold)  # Simulated result
    return indentation_score

def analyze_poster_size_and_dimension(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def analyze_image_clarity(image_path):
    img = Image.open(image_path)
    img = cvtColor(np.array(img), COLOR_BGR2GRAY)
    clarity_score = Laplacian(img, cv2.CV_64F).var()
    return clarity_score


def analyze_clutter(image_path):
    img = Image.open(image_path)
    img = cvtColor(np.array(img), COLOR_BGR2GRAY)

    # Use edge detection to identify edges in the image
    edges = Canny(img, threshold1=100, threshold2=200)

    # Count the number of edge pixels as a measure of clutter
    clutter_score = np.count_nonzero(edges)
    
    return clutter_score

@app.route("/", methods=["GET", "POST"])
def index():
    average_rgb_result = ""  # Initialize with an empty string
    indentation_result = ""
    size_dimension_result = ""
    clarity_result = ""
    clutter_result = ""
    qr_code_image = None
    if request.method == "POST":
        # Check if the post request has a file part
        if 'poster' not in request.files:
            return "No file part"

        file = request.files['poster']

        if file.filename == '':
            return "No selected file"

        if file:
            # Save the uploaded poster
            poster_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(poster_path)

            # Calculate average RGB values
            r, g, b = calculate_average_rgb(poster_path)
            average_rgb_result = f"Average RGB: ({r}, {g}, {b})"

            # Evaluate indentation (replace with actual model prediction)
            indentation_score = evaluate_indentation(poster_path)
            indentation_result = f"Indentation Evaluation: {indentation_score}"

            clarity_score = analyze_image_clarity(poster_path)
            clarity_result = f"Image Clarity Score: {clarity_score}"

            width, height = analyze_poster_size_and_dimension(poster_path)
            size_dimension_result = f"Poster Size: {width}x{height}"
            
            clutter_score = analyze_clutter(poster_path)
            clutter_result = f"Clutter Score: {clutter_score}"


            return f"{average_rgb_result}<br>{indentation_result}<br>{size_dimension_result}<br>{clarity_result}<br>{clutter_result}"

    return render_template('index.html', 
               average_rgb_result=average_rgb_result,
               indentation_result=indentation_result,
               size_dimension_result=size_dimension_result,
               clarity_result=clarity_result,
               clutter_result=clutter_result,
               show_results=True)

if __name__ == '__main__':
    app.run(debug=True)

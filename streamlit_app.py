import os
import streamlit as st
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import extract_v3
import categorize
from PIL import Image

# Set Streamlit Page Config
st.set_page_config(page_title="Handwriting Personality Predictor", layout="wide")

# Custom CSS for Better UI
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            color: #333333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stSidebar {
            background-color: #ffffff;
            border-right: 2px solid #dddddd;
        }
        .trait-yes {
            padding: 10px;
            border-radius: 5px;
            background-color: #155724;
            color: white;
            margin-bottom: 10px;
        }
        .trait-no {
            padding: 10px;
            border-radius: 5px;
            background-color: #721c24;
            color: white;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Settings
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Use this panel for navigation and settings.")

# Title
st.title("📝 AI-Powered Handwriting Personality Analysis")
st.write("🔍 Upload a handwriting sample to analyze personality traits.")

# Load Data
if os.path.isfile("label_list"):
    st.sidebar.success("✅ Data Loaded Successfully")

    X_baseline_angle, X_top_margin, X_letter_size = [], [], []
    X_line_spacing, X_word_spacing, X_pen_pressure, X_slant_angle = [], [], [], []
    y_t1, y_t2, y_t3, y_t4, y_t5, y_t6, y_t7, y_t8 = [], [], [], [], [], [], [], []
    page_ids = []

    with open("label_list", "r") as labels:
        for line in labels:
            content = line.split()
            X_baseline_angle.append(float(content[0]))
            X_top_margin.append(float(content[1]))
            X_letter_size.append(float(content[2]))
            X_line_spacing.append(float(content[3]))
            X_word_spacing.append(float(content[4]))
            X_pen_pressure.append(float(content[5]))
            X_slant_angle.append(float(content[6]))
            y_t1.append(float(content[7]))
            y_t2.append(float(content[8]))
            y_t3.append(float(content[9]))
            y_t4.append(float(content[10]))
            y_t5.append(float(content[11]))
            y_t6.append(float(content[12]))
            y_t7.append(float(content[13]))
            y_t8.append(float(content[14]))
            page_ids.append(content[15])

    # Feature Engineering
    features = [
        (X_baseline_angle, X_slant_angle),
        (X_letter_size, X_pen_pressure),
        (X_letter_size, X_top_margin),
        (X_line_spacing, X_word_spacing),
        (X_slant_angle, X_top_margin),
        (X_letter_size, X_line_spacing),
        (X_letter_size, X_word_spacing),
        (X_line_spacing, X_word_spacing)
    ]

    trait_labels = [
        "Emotional Stability",
        "Mental Energy or Will Power",
        "Modesty",
        "Personal Harmony and Flexibility",
        "Lack of Discipline",
        "Poor Concentration",
        "Non-Communicativeness",
        "Social Isolation"
    ]

    classifiers = []
    accuracies = []

    def train_and_save_classifier(X, y, filename):
        """Train and Save Classifier"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=8)
        clf = SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        joblib.dump(clf, filename)
        return clf, accuracy_score(clf.predict(X_test), y_test)

    # Train or Load Models
    if not all(os.path.exists(f"classifier_{i}.pkl") for i in range(1, 9)):
        for i, (X, y) in enumerate(zip(features, [y_t1, y_t2, y_t3, y_t4, y_t5, y_t6, y_t7, y_t8])):
            clf, acc = train_and_save_classifier(list(zip(*X)), y, f"classifier_{i+1}.pkl")
            classifiers.append(clf)
            accuracies.append(acc)
    else:
        for i in range(1, 9):
            classifiers.append(joblib.load(f"classifier_{i}.pkl"))

    # Display Classifier Accuracy in Sidebar
    st.sidebar.subheader("📊 Classifier Accuracy")
    for i, acc in enumerate(accuracies):
        st.sidebar.write(f"**{trait_labels[i]}**: {acc:.2f}")

    # Upload Image for Prediction
    uploaded_file = st.file_uploader("📤 Upload a Handwriting Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📌 Uploaded Handwriting Sample", width=250)

        if st.button("🔍 Predict Personality Traits"):
            image_path = "temp_image.jpg"
            image.save(image_path)

            # Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("⏳ Extracting Handwriting Features...")
            progress_bar.progress(25)

            # Extract Features
            raw_features = extract_v3.start(image_path)

            if raw_features is None:
                st.error("⚠️ Error: Failed to process image. Please try another one.")
            else:
                progress_bar.progress(50)
                status_text.text("🔍 Analyzing Handwriting Features...")

                # Categorize Features
                raw_baseline_angle, raw_top_margin, raw_letter_size = raw_features[:3]
                raw_line_spacing, raw_word_spacing, raw_pen_pressure, raw_slant_angle = raw_features[3:]

                st.subheader("📊 Extracted Handwriting Features")
                extracted_features = {
                    "Baseline Angle": raw_baseline_angle,
                    "Top Margin": raw_top_margin,
                    "Letter Size": raw_letter_size,
                    "Line Spacing": raw_line_spacing,
                    "Word Spacing": raw_word_spacing,
                    "Pen Pressure": raw_pen_pressure,
                    "Slant": raw_slant_angle
                }
                st.write(extracted_features)  # Debugging - Check if values change

                progress_bar.progress(75)
                status_text.text("🧠 Predicting Personality Traits...")

                # Predictions
                input_features = [
                    [raw_baseline_angle, raw_slant_angle],
                    [raw_letter_size, raw_pen_pressure],
                    [raw_letter_size, raw_top_margin],
                    [raw_line_spacing, raw_word_spacing],
                    [raw_slant_angle, raw_top_margin],
                    [raw_letter_size, raw_line_spacing],
                    [raw_letter_size, raw_word_spacing],
                    [raw_line_spacing, raw_word_spacing]
                ]
                
                predictions = {
                    trait_labels[i]: classifiers[i].predict([input_features[i]])[0]
                    for i in range(len(trait_labels))
                }

                # Function to Convert Boolean to "Yes" or "No"
                def bool_to_yes_no(value):
                    return "Yes" if value == 1 else "No"

                st.write("## 🧠 Predicted Personality Traits:")
                for trait, prediction in predictions.items():
                    color_class = "trait-yes" if bool_to_yes_no(prediction) == "Yes" else "trait-no"
                    st.markdown(f"<div class='{color_class}'><strong>{trait}:</strong> {bool_to_yes_no(prediction)}</div>", unsafe_allow_html=True)

                progress_bar.progress(100)
                status_text.text("✅ Prediction Completed!")
import os
import numpy as np
import itertools
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import extract_v3
import categorize

# Feature lists
X_baseline_angle = []
X_top_margin = []
X_letter_size = []
X_line_spacing = []
X_word_spacing = []
X_pen_pressure = []
X_slant_angle = []

# Label lists
y_t1, y_t2, y_t3, y_t4, y_t5, y_t6, y_t7, y_t8 = [], [], [], [], [], [], [], []
page_ids = []

if os.path.isfile("label_list"):
    print("Info: label_list found.")
    
    with open("label_list", "r") as labels:
        for line in labels:
            content = line.split()
            
            # Extract features
            X_baseline_angle.append(float(content[0]))
            X_top_margin.append(float(content[1]))
            X_letter_size.append(float(content[2]))
            X_line_spacing.append(float(content[3]))
            X_word_spacing.append(float(content[4]))
            X_pen_pressure.append(float(content[5]))
            X_slant_angle.append(float(content[6]))
            
            # Extract labels
            y_t1.append(float(content[7]))
            y_t2.append(float(content[8]))
            y_t3.append(float(content[9]))
            y_t4.append(float(content[10]))
            y_t5.append(float(content[11]))
            y_t6.append(float(content[12]))
            y_t7.append(float(content[13]))
            y_t8.append(float(content[14]))
            page_ids.append(content[15])

    # Debug: Check unique labels
    print("Unique values in y_t1:", set(y_t1))
    
    # Feature combinations for different traits
    X_t1 = list(zip(X_baseline_angle, X_slant_angle))
    X_t2 = list(zip(X_letter_size, X_pen_pressure))
    X_t3 = list(zip(X_letter_size, X_top_margin))
    X_t4 = list(zip(X_line_spacing, X_word_spacing))
    X_t5 = list(zip(X_slant_angle, X_top_margin))
    X_t6 = list(zip(X_letter_size, X_line_spacing))
    X_t7 = list(zip(X_letter_size, X_word_spacing))
    X_t8 = list(zip(X_line_spacing, X_word_spacing))

    # Scale features
    scaler = StandardScaler()
    X_t1, X_t2, X_t3, X_t4 = scaler.fit_transform(X_t1), scaler.fit_transform(X_t2), scaler.fit_transform(X_t3), scaler.fit_transform(X_t4)
    X_t5, X_t6, X_t7, X_t8 = scaler.fit_transform(X_t5), scaler.fit_transform(X_t6), scaler.fit_transform(X_t7), scaler.fit_transform(X_t8)

    # Function to train and evaluate a classifier
    def train_and_evaluate(X, y, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, shuffle=True)
        clf = SVC(kernel='linear')  # Use linear kernel to avoid overfitting
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(clf.predict(X_test), y_test)
        print(f"Classifier accuracy: {accuracy:.2f}")
        return clf

    # Train classifiers
    clf1 = train_and_evaluate(X_t1, y_t1, random_state=8)
    clf2 = train_and_evaluate(X_t2, y_t2, random_state=16)
    clf3 = train_and_evaluate(X_t3, y_t3, random_state=32)
    clf4 = train_and_evaluate(X_t4, y_t4, random_state=64)
    clf5 = train_and_evaluate(X_t5, y_t5, random_state=42)
    clf6 = train_and_evaluate(X_t6, y_t6, random_state=52)
    clf7 = train_and_evaluate(X_t7, y_t7, random_state=21)
    clf8 = train_and_evaluate(X_t8, y_t8, random_state=73)

    # Prediction loop
    while True:
        file_name = input("Enter file name to predict or z to exit: ")
        if file_name.lower() == 'z':
            break

        raw_features = extract_v3.start(file_name)

        raw_baseline_angle = raw_features[0]
        baseline_angle, comment = categorize.determine_baseline_angle(
            raw_baseline_angle)
        print("Baseline Angle: "+comment)

        raw_top_margin = raw_features[1]
        top_margin, comment = categorize.determine_top_margin(raw_top_margin)
        print("Top Margin: "+comment)

        raw_letter_size = raw_features[2]
        letter_size, comment = categorize.determine_letter_size(
            raw_letter_size)
        print("Letter Size: "+comment)

        raw_line_spacing = raw_features[3]
        line_spacing, comment = categorize.determine_line_spacing(
            raw_line_spacing)
        print("Line Spacing: "+comment)

        raw_word_spacing = raw_features[4]
        word_spacing, comment = categorize.determine_word_spacing(
            raw_word_spacing)
        print("Word Spacing: "+comment)

        raw_pen_pressure = raw_features[5]
        pen_pressure, comment = categorize.determine_pen_pressure(
            raw_pen_pressure)
        print("Pen Pressure: "+comment)

        raw_slant_angle = raw_features[6]
        slant_angle, comment = categorize.determine_slant_angle(
            raw_slant_angle)
        print("Slant: "+comment)

        print
        print("Emotional Stability: ", clf1.predict(
            [[baseline_angle, slant_angle]]))
        print("Mental Energy or Will Power: ",
              clf2.predict([[letter_size, pen_pressure]]))
        print("Modesty: ", clf3.predict([[letter_size, top_margin]]))
        print("Personal Harmony and Flexibility: ",
              clf4.predict([[line_spacing, word_spacing]]))
        print("Lack of Discipline: ", clf5.predict(
            [[slant_angle, top_margin]]))
        print("Poor Concentration: ", clf6.predict(
            [[letter_size, line_spacing]]))
        print("Non Communicativeness: ", clf7.predict(
            [[letter_size, word_spacing]]))
        print("Social Isolation: ", clf8.predict(
            [[line_spacing, word_spacing]]))
        print("---------------------------------------------------")

else:
    print("Error: label_list file not found.")

import os
import extract_v3

# Change to 'images' directory and collect all file names
image_dir = "images"
if not os.path.isdir(image_dir):
    print(f"Error: '{image_dir}' directory not found.")
    exit(1)

files = [entry.name for entry in os.scandir(image_dir) if entry.is_file()]
if not files:
    print("No image files found in 'images' directory.")
    exit(1)

# Read existing processed page IDs
page_ids = set()
feature_list_file = "raw_feature_list"

if os.path.isfile(feature_list_file):
    print("Info: 'raw_feature_list' already exists. Reading processed files...")
    with open(feature_list_file, "r") as label:
        for line in label:
            content = line.split()
            if content:
                page_ids.add(content[-1])  # Last item is the file name

# Open the feature list file in append mode
with open(feature_list_file, "a") as label:
    count = len(page_ids)

    for file_name in files:
        if file_name in page_ids:
            continue  # Skip already processed files

        # Extract features
        file_path = os.path.join(image_dir, file_name)
        features = extract_v3.start(file_path)

        if not features or not isinstance(features, list):
            print(f"Warning: No valid features extracted for {file_name}. Skipping...")
            continue

        features.append(file_name)  # Append filename at the end

        # Write extracted features to the file
        label.write("\t".join(map(str, features)) + "\n")

        label.write("\n")

        # Update progress
        count += 1
        progress = (count * 100) / len(files)
        print(f"{count}/{len(files)} processed - {file_name} ({progress:.2f}%)")

print("Processing complete!")

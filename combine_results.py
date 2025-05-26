import os
import re
import json
from tqdm import tqdm

def extract_json_objects_from_file(input_file_path):
    """
    Extract JSON objects from a file where each line may be a complete JSON object
    """
    json_objects = []

    try:
        with open(input_file_path, 'r') as file:
            content = file.read()

            # Use regex to find JSON-like structures (starting with { and ending with })
            pattern = r'(\{.*?\})'
            matches = re.findall(pattern, content, re.DOTALL)

            for potential_json in matches:
                try:
                    json_obj = json.loads(potential_json)

                    # Validate required fields
                    required_fields = ["index", "title", "abstract", "decision", "explanation"]

                    if all(field in json_obj for field in required_fields):
                        json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue  # Skip if not valid JSON
    except Exception as e:
        print(f"Error processing file {input_file_path}: {str(e)}")

    return json_objects

def main():
    print("Starting to combine results...")

    # Directory containing the output files
    output_dir = 'decision'

    # Get all files in the decision directory
    try:
        files = os.listdir(output_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{output_dir}' not found")
        return

    # Filter for files matching the pattern and extract file numbers
    pattern = re.compile(r'Screened_Papers_(\d+)\.txt')
    matching_files = []

    for f in files:
        match = pattern.match(f)
        if match:
            file_number = int(match.group(1))
            matching_files.append((file_number, os.path.join(output_dir, f)))

    if not matching_files:
        print("No matching files found to process")
        return

    # Sort the files based on their numbers
    matching_files.sort(key=lambda x: x[0])

    all_json_objects = []
    total_files = len(matching_files)

    # Extract JSON objects from each matching file and collect them
    for i, (file_number, file_name) in enumerate(matching_files, 1):
        print(f"\nProcessing file {i}/{total_files}: {file_name}")
        json_objects = extract_json_objects_from_file(file_name)
        all_json_objects.extend(json_objects)
        print(f"Extracted {len(json_objects)} JSON objects")

    if not all_json_objects:
        print("No valid JSON objects found in any file")
        return

    # Write all JSON objects into one combined JSON file
    output_file_path = os.path.join(output_dir, 'Combined_Aging_Screening.json')
    try:
        with open(output_file_path, 'w') as file:
            json.dump(all_json_objects, file, indent=2)

        print(f"\nSuccessfully merged all JSON objects into {output_file_path}")
        print(f"Total number of objects processed: {len(all_json_objects)}")

    except Exception as e:
        print(f"Error writing combined file: {str(e)}")

if __name__ == "__main__":
    main()
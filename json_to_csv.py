import json
import csv
import os
from tqdm import tqdm

def json_to_csv(json_path):
    try:
        # Read the JSON file
        print("Reading JSON file...")
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Define the output CSV path
        csv_path = os.path.splitext(json_path)[0] + '.csv'

        # Ensure data is a list
        if not isinstance(data, list):
            print("Error: JSON data should be an array of objects.")
            return

        # Gather all unique fields from every item
        print("Gathering unique fields...")
        all_fieldnames = set()
        for item in tqdm(data, desc="Processing fields"):
            if isinstance(item, dict):
                all_fieldnames.update(item.keys())

        fieldnames = list(all_fieldnames)

        # Write to CSV file
        print("Writing to CSV...")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for item in tqdm(data, desc="Writing rows"):
                writer.writerow(item)

        print(f"CSV file created successfully: {csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use the combined JSON file from the decision directory
    json_file_path = "decision/Combined_Aging_Screening.json"
    json_to_csv(json_file_path) 
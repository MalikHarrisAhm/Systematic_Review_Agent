import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://api.deepseek.com/"  # Added /v1 to the base URL

# Create directories for progress tracking and output
progress_dir = Path('progress_tracking')
progress_dir.mkdir(exist_ok=True)

output_dir = Path('decision')
output_dir.mkdir(exist_ok=True)

def load_screening_criteria():
    """Load screening criteria from file"""
    with open('screening_criteria.txt', 'r') as f:
        return json.load(f)

def update_progress(current, total):
    """Update progress in a file that can be read by the UI"""
    progress_file = progress_dir / 'screening_progress.json'
    with open(progress_file, 'w') as f:
        json.dump({
            'current': current,
            'total': total,
            'percentage': (current / total) * 100 if total > 0 else 0
        }, f)

def process_abstract(index, title, abstract, doi, criteria):
    """Process a single abstract using OpenAI API"""
    try:
        system_prompt = """
        Provide comprehensive and detailed responses to user inquiries. Ensure accuracy and completeness in information. 
        Verify facts when necessary. Maintain a neutral tone, and avoid personal opinions or biases. Be concise.
        """

        user_prompt = f"""
        Given the following abstract, determine whether it meets the following inclusion or exclusion criteria for a literature review. 
        Do not negotiate the criteria:

        INCLUSION CRITERIA:
        {chr(10).join(criteria['inclusion'])}

        EXCLUSION CRITERIA:
        {chr(10).join(criteria['exclusion'])}

        Index: {index}
        Title: {title}
        DOI: {doi}
        Abstract: {abstract}

        Output the decision in the following JSON format:
        {{
            "index": "{index}",
            "title": "{title}",
            "doi": "{doi}",
            "abstract": "{abstract}",
            "decision": "Include" or "Exclude",
            "explanation": "<Brief explanation>"
        }}
        """

        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=1500
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error processing abstract {index}: {str(e)}")
        return None

def main():
    print("Starting the screening process...")
    
    # Load screening criteria
    criteria = load_screening_criteria()
    
    # Read the CSV file
    print("Reading input CSV file...")
    df = pd.read_csv('pubmed_search_results.csv')
    
    # Take random sample of 10 if requested
    if os.path.exists('selected_model.txt'):
        with open('selected_model.txt', 'r') as f:
            if f.read().strip() == 'sample':
                if len(df) > 10:
                    print("Taking random sample of 10 papers...")
                    df = df.sample(n=10, random_state=42)
    
    total_papers = len(df)
    print(f"Processing {total_papers} papers...")
    
    # Initialize progress
    update_progress(0, total_papers)
    
    # Process abstracts in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(
                process_abstract,
                row.name,
                row['TI'],
                row['AB'],
                row['LID'],
                criteria
            )
            futures.append(future)
        
        # Collect results with progress bar
        results = []
        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing papers"):
            result = future.result()
            if result:
                results.append(result)
            completed += 1
            update_progress(completed, total_papers)
    
    print(f"\nProcessed {len(results)} papers successfully")
    
    # Save results
    print("Saving results...")
    output_file = output_dir / 'Screened_Papers_0.txt'
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + ',\n')
    
    # Convert to JSON and CSV
    print("Converting results to JSON and CSV...")
    json_objects = []
    for result in results:
        try:
            json_obj = json.loads(result)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue
    
    # Save combined JSON
    json_output = output_dir / 'Combined_Aging_Screening.json'
    with open(json_output, 'w') as f:
        json.dump(json_objects, f, indent=2)
    
    # Convert to CSV
    df_results = pd.DataFrame(json_objects)
    csv_output = output_dir / 'Combined_Aging_Screening.csv'
    df_results.to_csv(csv_output, index=False)
    
    # Update progress to 100%
    update_progress(total_papers, total_papers)
    
    print(f"\nResults saved to:")
    print(f"- {output_file}")
    print(f"- {json_output}")
    print(f"- {csv_output}")
    print("\nScreening process completed successfully!")

if __name__ == "__main__":
    main() 
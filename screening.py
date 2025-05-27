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
from openai import OpenAI
import sys

# Load environment variables
load_dotenv()

# Create directories for progress tracking and output
progress_dir = Path('progress_tracking')
progress_dir.mkdir(exist_ok=True)

output_dir = Path('decision')
output_dir.mkdir(exist_ok=True)

def get_model_info():
    """Read the selected model information from model_selection.json"""
    try:
        with open('model_selection.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default to OpenAI if no model selection file exists
        return {"provider": "OpenAI", "model": "gpt-4o"}

def get_client(model_info):
    """Get the appropriate OpenAI client based on the provider"""
    if model_info["provider"] == "OpenAI":
        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    else:  # DeepSeek
        return OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

def get_completion(prompt, model_info):
    """Get completion from the selected model"""
    client = get_client(model_info)
    
    # Map display names to API model names for DeepSeek
    model_mapping = {
        "Deepseek-chat (V3)": "deepseek-chat",
        "Deepseek-reasoner (R1)": "deepseek-reasoner"
    }
    
    # Get the correct model name
    api_model = model_mapping.get(model_info["model"], model_info["model"])
    
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": "You are a systematic review screening assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

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

def process_abstract(index, title, abstract, doi, criteria, model_info):
    """Process a single abstract using the selected model"""
    try:
        system_prompt = """
        You are a systematic review screening assistant. Your task is to evaluate abstracts based on inclusion and exclusion criteria.
        Provide your response in valid JSON format only, with no additional text.
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

        Output ONLY a valid JSON object in this exact format:
        {{
            "index": "{index}",
            "title": "{title}",
            "doi": "{doi}",
            "abstract": "{abstract}",
            "decision": "Include" or "Exclude",
            "explanation": "<Brief explanation>"
        }}
        """

        client = get_client(model_info)
        
        # Map display names to API model names for DeepSeek
        model_mapping = {
            "Deepseek-chat (V3)": "deepseek-chat",
            "Deepseek-reasoner (R1)": "deepseek-reasoner"
        }
        
        # Get the correct model name
        api_model = model_mapping.get(model_info["model"], model_info["model"])
        
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        
        # Clean and validate the response
        result = response.choices[0].message.content.strip()
        # Remove any markdown code block markers
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Validate JSON
        try:
            json_obj = json.loads(result)
            return json.dumps(json_obj)  # Return as a JSON string
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response for abstract {index}: {str(e)}")
            print(f"Raw response: {result}")
            return None

    except Exception as e:
        print(f"Error processing abstract {index}: {str(e)}")
        return None

def main():
    print("Starting the screening process...")
    
    try:
        # Ensure output directories exist
        output_dir = Path('decision')
        output_dir.mkdir(exist_ok=True)
        
        progress_dir = Path('progress_tracking')
        progress_dir.mkdir(exist_ok=True)
        
        # Initialize progress
        update_progress(0, 0)
        
        # Load screening criteria
        criteria = load_screening_criteria()
        
        # Get model information
        model_info = get_model_info()
        
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
        
        # Update progress with total count
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
                    criteria,
                    model_info
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
        
        if not results:
            print("No valid results were generated!")
            return False
        
        print(f"\nProcessed {len(results)} papers successfully")
        
        # Save results
        print("Saving results...")
        output_file = output_dir / 'Screened_Papers_0.txt'
        with open(output_file, 'w') as f:
            for result in results:
                f.write(result + '\n')  # Write each JSON object on a new line
        
        # Convert to JSON and CSV
        print("Converting results to JSON and CSV...")
        json_objects = []
        for result in results:
            try:
                json_obj = json.loads(result)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON result: {str(e)}")
                print(f"Problematic result: {result}")
                continue
        
        if not json_objects:
            print("No valid results to save!")
            return False
        
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
        return True
        
    except Exception as e:
        print(f"An error occurred during screening: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
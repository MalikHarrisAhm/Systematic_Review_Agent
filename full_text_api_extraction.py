import requests
import pandas as pd
import sys
import time
import signal
from tqdm import tqdm
from collections import defaultdict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables to track statistics
success_count = 0
fail_count = 0
source_stats = defaultdict(lambda: {'success': 0, 'attempts': 0})
total_rows = 0
start_time = None

# API Credentials from environment variables
API_KEYS = {
    'springer': os.getenv('SPRINGER_API_KEY'),
    'elsevier': os.getenv('ELSEVIER_API_KEY'),
    'wiley': os.getenv('WILEY_API_KEY')
}

def validate_api_keys():
    """Validate that all required API keys are present"""
    missing_keys = [k for k, v in API_KEYS.items() if not v]
    if missing_keys:
        print(f"Error: Missing API keys for: {', '.join(missing_keys)}")
        print("Please ensure you have set up your .env file with the following variables:")
        print("SPRINGER_API_KEY=your_springer_key")
        print("ELSEVIER_API_KEY=your_elsevier_key")
        print("WILEY_API_KEY=your_wiley_key")
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle interrupt signal and print statistics before exiting"""
    print("\n\nScript interrupted! Printing final statistics...")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print time statistics
    print(f"\nTime elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Print processing statistics
    processed_rows = success_count + fail_count
    print(f"\nProcessing Statistics:")
    print(f"Processed {processed_rows} out of {total_rows} papers ({(processed_rows / total_rows) * 100:.1f}%)")
    print(f"Success: {success_count} papers")
    print(f"Failed: {fail_count} papers")
    if processed_rows > 0:
        print(f"Success Rate: {(success_count / processed_rows) * 100:.2f}%")

    # Print source statistics
    print_source_statistics(source_stats, processed_rows)

    sys.exit(0)


def print_source_statistics(source_stats, total_attempts):
    """Print detailed statistics for each source"""
    print("\nSource Statistics:")
    print("-" * 80)
    print(f"{'Source':<15} {'Success':<10} {'Attempts':<10} {'Success Rate':<15} {'% of Total Success':<20}")
    print("-" * 80)

    total_successes = sum(stats['success'] for stats in source_stats.values())

    for source, stats in sorted(source_stats.items()):
        success_rate = (stats['success'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0
        total_success_contribution = (stats['success'] / total_successes * 100) if total_successes > 0 else 0

        print(f"{source:<15} {stats['success']:<10} {stats['attempts']:<10} "
              f"{success_rate:>6.1f}%{' ':>8} {total_success_contribution:>6.1f}%")
    print("-" * 80)


def get_text_from_ncbi(doi):
    """Retrieve text from NCBI using DOI"""
    tool_name = os.getenv('NCBI_TOOL_NAME', 'my_tool')
    email = os.getenv('NCBI_EMAIL', 'your_email@example.com')
    
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool={tool_name}&email={email}&ids={doi}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'records' in data and len(data['records']) > 0:
            pmcid = data['records'][0].get('pmcid')
            if pmcid:
                full_text_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&tool={tool_name}&email={email}"
                response = requests.get(full_text_url)
                if response.status_code == 200:
                    return response.text
    return None


def get_text_from_springer(doi):
    """Retrieve text from Springer using DOI"""
    if not API_KEYS['springer']:
        return None
    
    url = f"https://api.springernature.com/meta/v2/json?q=doi:{doi}&api_key={API_KEYS['springer']}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'records' in data and len(data['records']) > 0:
            return data['records'][0].get('fullText')
    return None


def get_text_from_elsevier(doi):
    """Retrieve text from Elsevier using DOI"""
    if not API_KEYS['elsevier']:
        return None
        
    headers = {'X-ELS-APIKey': API_KEYS['elsevier'], 'Accept': 'application/json'}
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get('full-text-retrieval-response', {}).get('coredata', {}).get('dc:description')
    return None


def get_text_from_wiley(doi):
    """Retrieve text from Wiley using DOI"""
    if not API_KEYS['wiley']:
        return None
        
    url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"
    headers = {'Authorization': f'Bearer {API_KEYS["wiley"]}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get('fullText')
    return None


def get_text_from_biorxiv(doi):
    """Retrieve text from bioRxiv using DOI"""
    url = f"https://api.biorxiv.org/details/biorxiv/{doi}/na/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'collection' in data and len(data['collection']) > 0:
            return data['collection'][0].get('fullText')
    return None


def get_text_from_medrxiv(doi):
    """Retrieve text from medRxiv using DOI"""
    url = f"https://api.medrxiv.org/details/medrxiv/{doi}/na/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'collection' in data and len(data['collection']) > 0:
            return data['collection'][0].get('fullText')
    return None


def get_text_from_arxiv(doi):
    """Retrieve text from arXiv using DOI"""
    arxiv_id = doi.split('/')[-1]
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200 and "<entry>" in response.text:
        return response.text
    return None


def process_csv_and_append(papers_file, output_file, start_row=0):
    global success_count, fail_count, source_stats, total_rows, start_time

    # Validate API keys before starting
    validate_api_keys()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Record start time
    start_time = time.time()

    # Read the CSV file
    papers_df = pd.read_csv(papers_file)

    # Initialize or create required columns
    if 'Full Text' not in papers_df.columns:
        papers_df['Full Text'] = ""
    if 'TXT or PDF' not in papers_df.columns:
        papers_df['TXT or PDF'] = ""

    # Set total rows for statistics
    total_rows = len(papers_df) - start_row

    # Create progress bar
    pbar = tqdm(
        total=total_rows,
        desc="Processing papers",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    # List of sources and their corresponding functions
    sources = [
        ("NCBI", get_text_from_ncbi),
        ("Springer", get_text_from_springer),
        ("Elsevier", get_text_from_elsevier),
        ("Wiley", get_text_from_wiley),
        ("Biorxiv", get_text_from_biorxiv),
        ("Medrxiv", get_text_from_medrxiv),
        ("arXiv", get_text_from_arxiv)
    ]

    try:
        # Process each paper
        for index, row in papers_df.iloc[start_row:].iterrows():
            doi = row['DOI']

            full_text = None
            # Try each source in sequence
            for source_name, source_function in sources:
                try:
                    source_stats[source_name]['attempts'] += 1
                    full_text = source_function(doi)
                    if full_text:
                        papers_df.at[index, 'Full Text'] = full_text
                        papers_df.at[index, 'TXT or PDF'] = source_name
                        source_stats[source_name]['success'] += 1
                        success_count += 1
                        break
                except Exception as e:
                    print(f"\nError processing {doi} with {source_name}: {str(e)}")
                    continue

            if not full_text:
                fail_count += 1

            # Update progress bar with current stats
            success_rate = (success_count / (success_count + fail_count) * 100) if (
                    success_count + fail_count) > 0 else 0
            pbar.set_description(
                f"Processing papers (Success: {success_count}, Failed: {fail_count}, "
                f"Success Rate: {success_rate:.1f}%)"
            )
            pbar.update(1)

            # Save progress periodically (every 10 papers)
            if (index - start_row + 1) % 10 == 0:
                papers_df.to_csv(output_file, index=False)

            # Rate limiting to avoid overwhelming APIs
            time.sleep(0.1)

    except KeyboardInterrupt:
        # The signal handler will take care of printing statistics
        pass
    finally:
        # Close progress bar and save final results
        pbar.close()
        papers_df.to_csv(output_file, index=False)

    # Print final summary if script completes normally
    print(f"\nProcessing complete!")
    print(f"Total Success: {success_count} papers")
    print(f"Total Failed: {fail_count} papers")
    print(f"Overall Success Rate: {(success_count / total_rows) * 100:.2f}%")

    # Print detailed source statistics
    print_source_statistics(source_stats, total_rows)


if __name__ == "__main__":
    # File paths - using relative paths for better portability
    papers_file = 'input/dimensions_dataset.csv'
    output_file = 'output/text_extraction.csv'
    start_row = 0

    # Run the main process
    process_csv_and_append(papers_file, output_file, start_row)

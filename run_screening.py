import os
import subprocess
import time

def run_script(script_name, description):
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"\n{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {str(e)}")
        return False

def main():
    print("Starting automated screening process...")
    
    # Create necessary directories
    os.makedirs('decision', exist_ok=True)
    os.makedirs('progress_tracking', exist_ok=True)
    
    # Run the screening script
    if not run_script('screening.py', 'Initial screening process'):
        print("Screening process failed. Stopping execution.")
        return
    
    # Run the combine results script
    if not run_script('combine_results.py', 'Combining results'):
        print("Combining results failed. Stopping execution.")
        return
    
    # Run the JSON to CSV conversion
    if not run_script('json_to_csv.py', 'Converting to CSV'):
        print("CSV conversion failed. Stopping execution.")
        return
    
    print("\nAll processes completed successfully!")
    print("\nResults are available in the 'decision' directory:")
    print("- Combined_Aging_Screening.json")
    print("- Combined_Aging_Screening.csv")

if __name__ == "__main__":
    main() 
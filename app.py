import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import openai
from dotenv import load_dotenv
import subprocess
import sys
from tqdm import tqdm
import requests
from Bio import Entrez, Medline
import plotly.express as px
import re
import matplotlib.pyplot as plt
import altair as alt
import ast
import time
import threading
from datetime import datetime
from fpdf import FPDF
import io
import base64

# Load environment variables
load_dotenv()

# Set page config with custom theme
st.set_page_config(
    page_title="Systematic Review Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-style dark theme
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background-color: #1c1c1e;
    }
    
    /* Text Styles */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Input Styles */
    .stTextInput>div>div>input {
        background-color: #1c1c1e;
        color: #ffffff;
        border: 1px solid #2c2c2e;
        border-radius: 8px;
    }
    
    .stTextArea>div>div>textarea {
        background-color: #1c1c1e;
        color: #ffffff;
        border: 1px solid #2c2c2e;
        border-radius: 8px;
    }
    
    /* Button Styles */
    .stButton>button {
        background-color: #0a84ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0071e3;
        box-shadow: 0 0 0 2px rgba(10, 132, 255, 0.3);
    }
    
    /* Select Box Styles */
    .stSelectbox>div>div {
        background-color: #1c1c1e;
        color: #ffffff;
        border: 1px solid #2c2c2e;
        border-radius: 8px;
    }
    
    /* Radio Button Styles */
    .stRadio>div {
        color: #ffffff;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div {
        background-color: #0a84ff;
    }
    
    /* Alert Styles */
    .stAlert {
        background-color: #1c1c1e;
        border: 1px solid #2c2c2e;
        border-radius: 8px;
    }
    
    /* Metric Styles */
    .stMetric {
        background-color: #1c1c1e;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #2c2c2e;
    }
    
    /* Dataframe Styles */
    .stDataFrame {
        background-color: #1c1c1e;
        border-radius: 8px;
        border: 1px solid #2c2c2e;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Reduce padding for main title */
    .main .block-container {
        padding-top: 1rem;
    }
    
    /* Links */
    a {
        color: #0a84ff;
        text-decoration: none;
    }
    
    a:hover {
        color: #0071e3;
    }
    
    /* Custom Container Styles */
    .custom-container {
        background-color: #1c1c1e;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid #2c2c2e;
    }
    
    /* Footer */
    .footer {
        color: #8e8e93;
        font-size: 0.8em;
        text-align: center;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS for small subtle toggle buttons
st.markdown('''
<style>
.small-toggle-btn button {
    font-size: 0.85em !important;
    padding: 2px 10px !important;
    border-radius: 6px !important;
    background: #222 !important;
    color: #aaa !important;
    border: 1px solid #333 !important;
    margin-left: 8px;
    margin-bottom: 0.5em;
}
.small-toggle-btn button:hover {
    background: #333 !important;
    color: #fff !important;
}
/* Reduce padding above each expander to bring it closer to the graph */
.stExpander {
    margin-top: -0.5em !important;
}
</style>
''', unsafe_allow_html=True)

# Initialize session state
if 'search_completed' not in st.session_state:
    st.session_state.search_completed = False
if 'screening_completed' not in st.session_state:
    st.session_state.screening_completed = False
if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

# Sidebar for API keys and model selection
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Keys Section
    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password")
    deepseek_key = st.text_input("DeepSeek API Key", type="password")
    
    if st.button("Save API Keys", use_container_width=True):
        if openai_key and deepseek_key:
            with open('.env', 'w') as f:
                f.write(f"OPENAI_API_KEY={openai_key}\n")
                f.write(f"DEEPSEEK_API_KEY={deepseek_key}\n")
            st.session_state.api_keys_set = True
            st.success("API keys saved successfully!")
        else:
            st.error("Please provide both API keys")
    
    # Model Selection
    st.subheader("Model Selection")
    model_provider = st.radio(
        "Select Model Provider",
        ["OpenAI", "DeepSeek"],
        help="Choose which provider's models to use"
    )
    
    if model_provider == "OpenAI":
        model = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4", "gpt-3.5-turbo"],
            help="Choose the OpenAI model to use"
        )
    else:
        model = st.selectbox(
            "Select DeepSeek Model",
            ["deepseek-chat", "deepseek-coder"],
            help="Choose the DeepSeek model to use"
        )
    
    st.session_state.model = model
    st.session_state.model_provider = model_provider

def suggest_search_terms(topic):
    """Use selected model to suggest PubMed search terms"""
    try:
        if st.session_state.model_provider == "OpenAI":
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model=st.session_state.model,
                messages=[
                    {"role": "system", "content": "You are a systematic review expert. Generate PubMed search terms for the given research topic."},
                    {"role": "user", "content": f"Generate PubMed search terms for: {topic}"}
                ]
            )
            return response.choices[0].message.content
        else:
            # Use requests for DeepSeek
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": st.session_state.model,
                "messages": [
                    {"role": "system", "content": "You are a systematic review expert. Generate PubMed search terms for the given research topic."},
                    {"role": "user", "content": f"Generate PubMed search terms for: {topic}"}
                ],
                "stream": False
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error generating search terms: {str(e)}")
        return None

def run_pubmed_search(search_terms):
    """Run the PubMed search in Python using Bio.Entrez, support batching for large result sets (up to 50,000)."""
    try:
        Entrez.email = os.getenv('ENTREZ_EMAIL', 'your_email@example.com')
        api_key = os.getenv('ENTREZ_API_KEY')
        # Step 1: Get total count
        search_handle = Entrez.esearch(db="pubmed", term=search_terms, retmax=0, api_key=api_key)
        search_results = Entrez.read(search_handle)
        total_count = int(search_results['Count'])
        max_results = min(total_count, 50000)
        batch_size = 1000
        all_ids = []
        # Step 2: Fetch IDs in batches
        for start in range(0, max_results, batch_size):
            handle = Entrez.esearch(
                db="pubmed",
                term=search_terms,
                retmax=batch_size,
                retstart=start,
                api_key=api_key
            )
            record = Entrez.read(handle)
            all_ids.extend(record["IdList"])
        # Step 3: Fetch details for all IDs in batches
        records = []
        for i in range(0, len(all_ids), 500):
            batch_ids = all_ids[i:i+500]
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch_ids),
                rettype="medline",
                retmode="text",
                api_key=api_key
            )
            batch_records = list(Medline.parse(handle))
            records.extend(batch_records)
        # Step 4: Convert to DataFrame and Save
        df = pd.DataFrame(records)
        df.to_csv("pubmed_search_results.csv", index=False)
        st.success(f"Found {len(all_ids)} articles. Results saved.")
        # Split author names by semicolon or comma
        if 'AU' in df.columns:
            def split_authors(val):
                if pd.isnull(val):
                    return val
                # Try splitting by semicolon first, then comma
                if ';' in str(val):
                    return [a.strip() for a in str(val).split(';') if a.strip()]
                elif ',' in str(val):
                    return [a.strip() for a in str(val).split(',') if a.strip()]
                else:
                    return [val] if val else []
            df['AU'] = df['AU'].apply(split_authors)
        return True, df
    except Exception as e:
        st.error(f"Error running PubMed search: {type(e).__name__}: {str(e)}\n\nCheck your query format and internet connection.")
        return False, None

def run_screening(criteria):
    """Run the screening script"""
    try:
        # Save criteria to a temporary file
        with open('screening_criteria.txt', 'w') as f:
            f.write(json.dumps(criteria))
        
        # Run the screening script
        subprocess.run(['python', 'screening.py'], check=True)
        return True
    except Exception as e:
        st.error(f"Error running screening: {str(e)}")
        return False

def get_screening_progress():
    """Get the current screening progress"""
    try:
        progress_file = Path('progress_tracking/screening_progress.json')
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'current': 0, 'total': 0, 'percentage': 0}

def generate_pdf_report(search_terms, df):
    """Generate a PDF report with search details and plots"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "PubMed Search Report", ln=True, align="C")
    pdf.ln(5)
    
    # Search Details
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Search Details", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Search Terms: {search_terms}", ln=True)
    pdf.cell(0, 10, f"Total Papers Found: {len(df)}", ln=True)
    pdf.ln(5)
    
    # Cost Breakdown
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Estimated Cost Breakdown", ln=True)
    pdf.set_font("Arial", "", 10)
    
    # Calculate tokens and costs
    try:
        import tiktoken
        enc_gpt = tiktoken.get_encoding("cl100k_base")
        def true_tokens(text):
            if not isinstance(text, str):
                return 0
            return len(enc_gpt.encode(text))
    except ImportError:
        def true_tokens(text):
            if not isinstance(text, str):
                return 0
            return max(1, int(len(text) / 4))
    
    total_tokens_gpt = (df['TI'].fillna('').apply(true_tokens) + df['AB'].fillna('').apply(true_tokens)).sum()
    total_tokens_deepseek = (df['TI'].fillna('').apply(lambda x: max(1, int(len(x)/4))) + df['AB'].fillna('').apply(lambda x: max(1, int(len(x)/4)))).sum()
    
    model_prices = [
        {"Model": "gpt-3.5-turbo", "Price": 0.0005, "Tokens": total_tokens_gpt},
        {"Model": "gpt-4", "Price": 0.01, "Tokens": total_tokens_gpt},
        {"Model": "deepseek-chat", "Price": 0.0002, "Tokens": total_tokens_deepseek},
        {"Model": "deepseek-coder", "Price": 0.0002, "Tokens": total_tokens_deepseek}
    ]
    
    # Add cost table
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 10, "Model", 1)
    pdf.cell(40, 10, "Tokens", 1)
    pdf.cell(40, 10, "Cost (USD)", 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 10)
    for model in model_prices:
        cost = 2 * model["Price"] * (model["Tokens"] / 1000)
        pdf.cell(60, 10, model["Model"], 1)
        pdf.cell(40, 10, f"{model['Tokens']:,}", 1)
        pdf.cell(40, 10, f"${cost:.2f}", 1)
        pdf.ln()
    
    pdf.ln(5)
    
    # Check if we need a new page for the first plot
    if pdf.get_y() > 200:
        pdf.add_page()
    
    # Generate and save plots
    # Year Distribution
    if 'EDAT' in df.columns:
        df['Year'] = df['EDAT'].astype(str).str[:4]
        year_df = df['Year'].value_counts().sort_index().reset_index()
        year_df.columns = ['Year', 'Count']
        year_df = year_df[year_df['Year'].str.isnumeric()]
        
        plt.figure(figsize=(8, 4))
        plt.bar(year_df['Year'], year_df['Count'])
        plt.title('Publication Year Distribution')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150)
        img_bytes.seek(0)
        plt.close()
        
        # Add plot to PDF
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Publication Year Distribution", ln=True)
        pdf.image(img_bytes, x=10, y=None, w=190)
        
        # Add small table
        pdf.set_font("Arial", "B", 10)
        pdf.cell(40, 10, "Year", 1)
        pdf.cell(40, 10, "Count", 1)
        pdf.ln()
        
        pdf.set_font("Arial", "", 10)
        for _, row in year_df.head(5).iterrows():
            pdf.cell(40, 10, str(row['Year']), 1)
            pdf.cell(40, 10, str(row['Count']), 1)
            pdf.ln()
        pdf.ln(5)
    
    # Check if we need a new page for the next plot
    if pdf.get_y() > 200:
        pdf.add_page()
    
    # Top Journals
    if 'JT' in df.columns:
        journal_df = df['JT'].value_counts().head(10).reset_index()
        journal_df.columns = ['Journal', 'Count']
        
        plt.figure(figsize=(8, 4))
        plt.barh(journal_df['Journal'], journal_df['Count'])
        plt.title('Top 10 Journals')
        plt.xlabel('Count')
        plt.ylabel('Journal')
        plt.tight_layout()
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150)
        img_bytes.seek(0)
        plt.close()
        
        # Add plot to PDF
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Top 10 Journals", ln=True)
        pdf.image(img_bytes, x=10, y=None, w=190)
        
        # Add small table
        pdf.set_font("Arial", "B", 10)
        pdf.cell(140, 10, "Journal", 1)
        pdf.cell(40, 10, "Count", 1)
        pdf.ln()
        
        pdf.set_font("Arial", "", 10)
        for _, row in journal_df.head(5).iterrows():
            pdf.cell(140, 10, str(row['Journal'])[:60], 1)
            pdf.cell(40, 10, str(row['Count']), 1)
            pdf.ln()
        pdf.ln(5)
    
    # Check if we need a new page for the next plot
    if pdf.get_y() > 200:
        pdf.add_page()
    
    # Top Authors
    if 'AU' in df.columns:
        def parse_au(val):
            if pd.isnull(val):
                return []
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, str):
                    return [parsed]
                else:
                    return list(parsed)
            except Exception:
                return [str(val)]
        
        authors = df['AU'].apply(parse_au).explode()
        authors = authors[authors.notnull() & (authors != '')]
        author_df = authors.value_counts().head(10).reset_index()
        author_df.columns = ['Author', 'Count']
        
        plt.figure(figsize=(8, 4))
        plt.barh(author_df['Author'], author_df['Count'])
        plt.title('Top 10 Authors')
        plt.xlabel('Count')
        plt.ylabel('Author')
        plt.tight_layout()
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150)
        img_bytes.seek(0)
        plt.close()
        
        # Add plot to PDF
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Top 10 Authors", ln=True)
        pdf.image(img_bytes, x=10, y=None, w=190)
        
        # Add small table
        pdf.set_font("Arial", "B", 10)
        pdf.cell(140, 10, "Author", 1)
        pdf.cell(40, 10, "Count", 1)
        pdf.ln()
        
        pdf.set_font("Arial", "", 10)
        for _, row in author_df.head(5).iterrows():
            pdf.cell(140, 10, str(row['Author'])[:60], 1)
            pdf.cell(40, 10, str(row['Count']), 1)
            pdf.ln()
        pdf.ln(5)
    
    # Check if we need a new page for the next plot
    if pdf.get_y() > 200:
        pdf.add_page()
    
    # Keyword Analysis
    if 'search_terms' in st.session_state:
        query = st.session_state['search_terms']
        keywords = re.findall(r'([\w\s\-]+)(?:\[.*?\])?', query)
        keywords = [k.strip() for k in keywords if k.strip() and k.strip().lower() not in ['and', 'or', 'not'] and len(k.strip()) > 2]
        
        if 'TI' in df.columns or 'AB' in df.columns:
            data = []
            for kw in keywords:
                title_count = df['TI'].astype(str).str.contains(kw, case=False, na=False).sum() if 'TI' in df.columns else 0
                abstract_count = df['AB'].astype(str).str.contains(kw, case=False, na=False).sum() if 'AB' in df.columns else 0
                data.append({'Keyword': kw, 'Section': 'Title', 'Count': title_count})
                data.append({'Keyword': kw, 'Section': 'Abstract', 'Count': abstract_count})
            
            keyword_df = pd.DataFrame(data)
            
            plt.figure(figsize=(8, 4))
            pivot_df = keyword_df.pivot(index='Keyword', columns='Section', values='Count')
            pivot_df.plot(kind='bar', stacked=True)
            plt.title('Keyword Occurrence in Titles and Abstracts')
            plt.xlabel('Keyword')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to bytes
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', dpi=150)
            img_bytes.seek(0)
            plt.close()
            
            # Add plot to PDF
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Keyword Analysis", ln=True)
            pdf.image(img_bytes, x=10, y=None, w=190)
            
            # Add small table
            pdf.set_font("Arial", "B", 10)
            pdf.cell(80, 10, "Keyword", 1)
            pdf.cell(50, 10, "Title Count", 1)
            pdf.cell(50, 10, "Abstract Count", 1)
            pdf.ln()
            
            pdf.set_font("Arial", "", 10)
            for kw in keywords:
                title_count = df['TI'].astype(str).str.contains(kw, case=False, na=False).sum()
                abstract_count = df['AB'].astype(str).str.contains(kw, case=False, na=False).sum()
                pdf.cell(80, 10, str(kw)[:30], 1)
                pdf.cell(50, 10, str(title_count), 1)
                pdf.cell(50, 10, str(abstract_count), 1)
                pdf.ln()
            pdf.ln(5)
    
    # Save PDF
    pdf_path = "pubmed_search_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Main app
st.title("🔍 Systematic Review AI Agent")

# Check if API keys are set
if not st.session_state.api_keys_set:
    st.warning("Please set your API keys in the sidebar to continue.")
    st.stop()

# Step 1: PubMed Search Query
with st.container():
    st.header("📝 Step 1: PubMed Search Query")
    
    # Add option to upload CSV or use PubMed
    search_source = st.radio(
        "Choose your data source:",
        ("PubMed Search", "Upload CSV File"),
        horizontal=True,
        index=0
    )
    
    if search_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                original_columns = df.columns.tolist()
                
                # Define column mapping (case-insensitive)
                column_mapping = {
                    'title': 'TI',
                    'abstract': 'AB',
                    'authors': 'AU',
                    'fulljournalname': 'JT',
                    'pubdate': 'EDAT',
                    'epubdate': 'EDAT',  # fallback if pubdate missing
                    'doi': 'LID',
                }
                # Lowercase for matching
                lower_cols = {col.lower(): col for col in df.columns}
                # Map columns
                mapped = {}
                for src, tgt in column_mapping.items():
                    if src in lower_cols:
                        mapped[tgt] = lower_cols[src]
                # Special handling for EDAT (prefer pubdate over epubdate)
                if 'pubdate' in lower_cols:
                    mapped['EDAT'] = lower_cols['pubdate']
                elif 'epubdate' in lower_cols:
                    mapped['EDAT'] = lower_cols['epubdate']
                # Rename columns
                df = df.rename(columns={v: k for k, v in mapped.items()})
                # Ensure all required columns exist
                required = ['TI', 'AB', 'AU', 'JT', 'EDAT', 'LID']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    st.error(f"Missing required columns after mapping: {missing}. Please check your CSV.")
                    st.stop()
                # Split author names by semicolon or comma
                if 'AU' in df.columns:
                    def split_authors(val):
                        if pd.isnull(val):
                            return val
                        # Try splitting by semicolon first, then comma
                        if ';' in str(val):
                            return [a.strip() for a in str(val).split(';') if a.strip()]
                        elif ',' in str(val):
                            return [a.strip() for a in str(val).split(',') if a.strip()]
                        else:
                            return [val] if val else []
                    df['AU'] = df['AU'].apply(split_authors)
                # Save the processed file
                df.to_csv("pubmed_search_results.csv", index=False)
                # Update session state
                st.session_state.search_completed = True
                st.session_state.pubmed_df = df
                st.session_state.search_terms = "Uploaded CSV File"
                st.success("CSV file processed and mapped successfully!")
                # Show column mapping results
                with st.expander("Column Mapping Results", expanded=True):
                    st.write("Original columns and their mapped names:")
                    mapping_df = pd.DataFrame({
                        'Original Column': original_columns,
                        'Mapped Column': [df.columns[i] if i < len(df.columns) else '' for i in range(len(original_columns))]
                    })
                    st.dataframe(mapping_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
    
    else:
        # Option to use model or manual entry, manual as default
        search_mode = st.radio(
            "How would you like to provide your PubMed search query?",
            ("Enter manually", "AI Generated (beta)"),
            horizontal=True,
            index=0
        )

        if search_mode == "AI Generated (beta)":
            topic = st.text_area(
                "Enter your research topic:",
                height=100,
                placeholder="Example: Effects of exercise on cognitive function in older adults"
            )
            if topic:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Generate Search Terms", use_container_width=True):
                        with st.spinner("Generating search terms..."):
                            search_terms = suggest_search_terms(topic)
                            if search_terms:
                                st.session_state.search_terms = search_terms
            if "search_terms" in st.session_state:
                st.text_area(
                    "Suggested Search Terms:",
                    st.session_state.search_terms,
                    height=200,
                    key="search_terms_display"
                )
                if st.button("Run PubMed Search", use_container_width=True, key="run_pubmed_search_ai"):
                    with st.spinner("Running PubMed search..."):
                        success, df = run_pubmed_search(st.session_state.search_terms)
                        if success:
                            st.session_state.search_completed = True
                            st.session_state.pubmed_df = df
                            st.success("PubMed search completed!")
                        else:
                            st.session_state.search_completed = False
        else:
            st.markdown("""
            **PubMed Query Format Example:**
            - Combine terms with AND/OR/NOT, use field tags like [Title/Abstract].
            - Example: `(exercise[Title/Abstract]) AND (cognitive function[Title/Abstract]) AND (older adults[Title/Abstract])`
            - See [PubMed Advanced Search Guide](https://pubmed.ncbi.nlm.nih.gov/advanced/) for more info.
            """)
            manual_query = st.text_area(
                "Enter your PubMed search query:",
                height=200,
                placeholder="Example: (exercise[Title/Abstract]) AND (cognitive function[Title/Abstract]) AND (older adults[Title/Abstract])"
            )
            if st.button("Run PubMed Search", use_container_width=True, key="run_pubmed_search_manual"):
                if manual_query.strip():
                    st.session_state.search_terms = manual_query
                    with st.spinner("Running PubMed search..."):
                        success, df = run_pubmed_search(manual_query)
                        if success:
                            st.session_state.search_completed = True
                            st.session_state.pubmed_df = df
                            st.success("PubMed search completed!")
                        else:
                            st.session_state.search_completed = False
                else:
                    st.error("Please enter a PubMed search query.")

# Step 1.5: Show CSV download and graphs if search completed
if st.session_state.get('search_completed'):
    st.header("📊 PubMed Search Results Overview")
    # Always read the real CSV file for analysis
    try:
        df = pd.read_csv("pubmed_search_results.csv")
        st.session_state.pubmed_df = df
    except Exception as e:
        st.error(f"Could not read results CSV: {e}")
        df = None
    if df is not None:
        # Add PDF Export button
        if st.button("📄 Export Search Report as PDF", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_path = generate_pdf_report(st.session_state.search_terms, df)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="pubmed_search_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # --- Preview Table Expander ---
        if not df.empty:
            with st.expander('Preview Table (First 5 Rows)', expanded=False):
                st.dataframe(df.head(), use_container_width=True, hide_index=True)
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="pubmed_search_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        # --- Publication Year Distribution ---
        if 'EDAT' in df.columns:
            df['Year'] = df['EDAT'].astype(str).str[:4]
            year_df = df['Year'].value_counts().sort_index().reset_index()
            year_df.columns = ['Year', 'Count']
            year_df = year_df[year_df['Year'].str.isnumeric()]
            if not year_df.empty:
                year_df['YearTrunc'] = year_df['Year']  # No truncation needed for years
                st.markdown('<h3 style="font-size:1.7em;font-weight:700;margin-top:1.2em;margin-bottom:1.2em;">Publication Year Distribution</h3>', unsafe_allow_html=True)
                chart = alt.Chart(year_df).mark_bar(color='#0a84ff').encode(
                    x=alt.X('YearTrunc:O', sort=None, axis=alt.Axis(labelAngle=35, labelLimit=1000, tickCount=len(year_df), labelOverlap=False, title='Year')),
                    y='Count:Q',
                    tooltip=['Year', 'Count']
                ).configure_view(
                    fill='transparent'
                ).configure_axis(
                    grid=False,
                    domain=False,
                    labelColor='#fff',
                    titleColor='#fff',
                    ticks=False,
                    tickColor='transparent',
                    tickBand='extent'
                ).configure_header(
                    labelColor='#fff',
                    titleColor='#fff',
                    labelFontSize=13,
                    titleFontSize=14,
                    labelFontWeight='normal'
                ).configure(
                    background='transparent'
                )
                st.altair_chart(chart, use_container_width=True)
                with st.expander('Year Table', expanded=False):
                    st.dataframe(year_df, use_container_width=True, hide_index=True)
        # --- Top Journals ---
        if 'JT' in df.columns:
            journal_df = df['JT'].value_counts().head(10).reset_index()
            journal_df.columns = ['Journal', 'Count']
            journal_df['JournalTrunc'] = journal_df['Journal'].apply(lambda x: x[:16] + '…' if len(x) > 16 else x)
            if not journal_df.empty:
                st.markdown('<h3 style="font-size:1.7em;font-weight:700;margin-top:1.2em;margin-bottom:1.2em;">Top 10 Journals</h3>', unsafe_allow_html=True)
                chart = alt.Chart(journal_df).mark_bar(color='#30d158').encode(
                    x=alt.X('JournalTrunc:N', sort='-y', axis=alt.Axis(labelAngle=35, labelLimit=1000, tickCount=len(journal_df), labelOverlap=False, title='Journal')),
                    y='Count:Q',
                    tooltip=['Journal', 'Count']
                ).configure_view(
                    fill='transparent'
                ).configure_axis(
                    grid=False,
                    domain=False,
                    labelColor='#fff',
                    titleColor='#fff',
                    ticks=False,
                    tickColor='transparent',
                    tickBand='extent'
                ).configure_header(
                    labelColor='#fff',
                    titleColor='#fff',
                    labelFontSize=13,
                    titleFontSize=14,
                    labelFontWeight='normal'
                ).configure(
                    background='transparent'
                )
                st.altair_chart(chart, use_container_width=True)
                with st.expander('Journal Table', expanded=False):
                    st.dataframe(journal_df, use_container_width=True, hide_index=True)
        # --- Top Authors ---
        if 'AU' in df.columns:
            def parse_au(val):
                if pd.isnull(val):
                    return []
                try:
                    # Try to parse as a list
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, str):
                        return [parsed]
                    else:
                        return list(parsed)
                except Exception:
                    # If not a list, treat as a single author string
                    return [str(val)]
            authors = df['AU'].apply(parse_au).explode()
            authors = authors[authors.notnull() & (authors != '')]
            author_df = authors.value_counts().head(10).reset_index()
            author_df.columns = ['Author', 'Count']
            author_df['AuthorTrunc'] = author_df['Author'].apply(lambda x: x[:16] + '…' if len(x) > 16 else x)
            if not author_df.empty:
                st.markdown('<h3 style="font-size:1.7em;font-weight:700;margin-top:1.2em;margin-bottom:1.2em;">Top 10 Authors</h3>', unsafe_allow_html=True)
                chart = alt.Chart(author_df).mark_bar(color='#ff375f').encode(
                    x=alt.X('AuthorTrunc:N', sort='-y', axis=alt.Axis(labelAngle=35, labelLimit=1000, tickCount=len(author_df), labelOverlap=False, title='Author')),
                    y='Count:Q',
                    tooltip=['Author', 'Count']
                ).configure_view(
                    fill='transparent'
                ).configure_axis(
                    grid=False,
                    domain=False,
                    labelColor='#fff',
                    titleColor='#fff',
                    ticks=False,
                    tickColor='transparent',
                    tickBand='extent'
                ).configure_header(
                    labelColor='#fff',
                    titleColor='#fff',
                    labelFontSize=13,
                    titleFontSize=14,
                    labelFontWeight='normal'
                ).configure(
                    background='transparent'
                )
                st.altair_chart(chart, use_container_width=True)
                with st.expander('Author Table', expanded=False):
                    st.dataframe(author_df, use_container_width=True, hide_index=True)
        # --- Keyword Stacked Bar Chart ---
        st.markdown('<h3 style="font-size:1.7em;font-weight:700;margin-top:1.2em;margin-bottom:1.2em;">Keyword Occurrence in Titles and Abstracts</h3>', unsafe_allow_html=True)
        if 'search_terms' in st.session_state:
            query = st.session_state['search_terms']
            keywords = re.findall(r'([\w\s\-]+)(?:\[.*?\])?', query)
            keywords = [k.strip() for k in keywords if k.strip() and k.strip().lower() not in ['and', 'or', 'not'] and len(k.strip()) > 2]
            if 'TI' in df.columns or 'AB' in df.columns:
                data = []
                for kw in keywords:
                    title_count = df['TI'].astype(str).str.contains(kw, case=False, na=False).sum() if 'TI' in df.columns else 0
                    abstract_count = df['AB'].astype(str).str.contains(kw, case=False, na=False).sum() if 'AB' in df.columns else 0
                    data.append({'Keyword': kw, 'Section': 'Title', 'Count': title_count})
                    data.append({'Keyword': kw, 'Section': 'Abstract', 'Count': abstract_count})
                keyword_df = pd.DataFrame(data)
                keyword_df['KeywordTrunc'] = keyword_df['Keyword'].apply(lambda x: x[:16] + '…' if len(x) > 16 else x)
                if not keyword_df.empty:
                    chart = alt.Chart(keyword_df).mark_bar().encode(
                        x=alt.X('KeywordTrunc:N', sort=None, axis=alt.Axis(labelAngle=35, labelLimit=1000, tickCount=len(keyword_df['Keyword'].unique()), labelOverlap=False, title='Keyword')),
                        y=alt.Y('Count:Q', stack='zero'),
                        color=alt.Color('Section:N', scale=alt.Scale(domain=['Title', 'Abstract'], range=['#0a84ff', '#30d158'])),
                        tooltip=['Keyword', 'Section', 'Count']
                    ).configure_view(
                        fill='transparent'
                    ).configure_axis(
                        grid=False,
                        domain=False,
                        labelColor='#fff',
                        titleColor='#fff',
                        ticks=False,
                        tickColor='transparent',
                        tickBand='extent'
                    ).configure_header(
                        labelColor='#fff',
                        titleColor='#fff',
                        labelFontSize=13,
                        titleFontSize=14,
                        labelFontWeight='normal'
                    ).configure(
                        background='transparent'
                    )
                    st.altair_chart(chart, use_container_width=True)
                    with st.expander('Keyword Table', expanded=False):
                        st.dataframe(keyword_df, use_container_width=True, hide_index=True)
        # --- Country Map (as bar chart for now) ---
        if 'PL' in df.columns:
            country_df = df['PL'].value_counts().reset_index()
            country_df.columns = ['Country', 'Count']
            country_df['CountryTrunc'] = country_df['Country'].apply(lambda x: x[:16] + '…' if len(x) > 16 else x)
            if not country_df.empty:
                st.markdown('<h3 style="font-size:1.7em;font-weight:700;margin-top:1.2em;margin-bottom:1.2em;">Number of Publications by Country</h3>', unsafe_allow_html=True)
                chart = alt.Chart(country_df).mark_bar(color='#0a84ff').encode(
                    x=alt.X('CountryTrunc:N', sort='-y', axis=alt.Axis(labelAngle=35, labelLimit=1000, tickCount=len(country_df), labelOverlap=False, title='Country')),
                    y='Count:Q',
                    tooltip=['Country', 'Count']
                ).configure_view(
                    fill='transparent'
                ).configure_axis(
                    grid=False,
                    domain=False,
                    labelColor='#fff',
                    titleColor='#fff',
                    ticks=False,
                    tickColor='transparent',
                    tickBand='extent'
                ).configure_header(
                    labelColor='#fff',
                    titleColor='#fff',
                    labelFontSize=13,
                    titleFontSize=14,
                    labelFontWeight='normal'
                ).configure(
                    background='transparent'
                )
                st.altair_chart(chart, use_container_width=True)
                with st.expander('Country Table', expanded=False):
                    st.dataframe(country_df, use_container_width=True, hide_index=True)

# Step 2: Inclusion/Exclusion Criteria
if st.session_state.search_completed:
    with st.container():
        st.header("🎯 Step 2: Define Inclusion/Exclusion Criteria")
        
        col1, col2 = st.columns(2)
        with col1:
            inclusion_criteria = st.text_area(
                "Enter inclusion criteria (one per line):",
                height=150,
                placeholder="Example:\n- Original research articles\n- Human studies\n- English language"
            )
            # Real-time validation for inclusion criteria
            if inclusion_criteria.strip():
                bad_lines = [i+1 for i, line in enumerate(inclusion_criteria.split('\n')) if line.strip() and not line.strip().startswith('- ')]
                if bad_lines:
                    st.warning(f"Lines {', '.join(map(str, bad_lines))} do not start with '- '. Please use '- ...' for each line.")
        with col2:
            exclusion_criteria = st.text_area(
                "Enter exclusion criteria (one per line):",
                height=150,
                placeholder="Example:\n- Review articles\n- Animal studies\n- Non-English language"
            )
            # Real-time validation for exclusion criteria
            if exclusion_criteria.strip():
                bad_lines = [i+1 for i, line in enumerate(exclusion_criteria.split('\n')) if line.strip() and not line.strip().startswith('- ')]
                if bad_lines:
                    st.warning(f"Lines {', '.join(map(str, bad_lines))} do not start with '- '. Please use '- ...' for each line.")
        
        # --- Screening Price Estimation Graph (now updates in real time) ---
        try:
            import tiktoken
            enc_gpt = tiktoken.get_encoding("cl100k_base")
            def true_tokens(text):
                if not isinstance(text, str):
                    return 0
                return len(enc_gpt.encode(text))
        except ImportError:
            def true_tokens(text):
                if not isinstance(text, str):
                    return 0
                return max(1, int(len(text) / 4))
        total_tokens_gpt = 0
        total_tokens_deepseek = 0
        n_articles = 0
        if st.session_state.get('pubmed_df') is not None:
            df = st.session_state['pubmed_df']
            n_articles = len(df)
            if 'TI' in df.columns and 'AB' in df.columns:
                total_tokens_gpt = (df['TI'].fillna('').apply(true_tokens) + df['AB'].fillna('').apply(true_tokens)).sum()
                total_tokens_deepseek = (df['TI'].fillna('').apply(lambda x: max(1, int(len(x)/4))) + df['AB'].fillna('').apply(lambda x: max(1, int(len(x)/4)))).sum()
        crit_tokens_gpt = (true_tokens(inclusion_criteria) + true_tokens(exclusion_criteria)) * max(1, n_articles)
        crit_tokens_deepseek = (max(1, int(len(inclusion_criteria)/4)) + max(1, int(len(exclusion_criteria)/4))) * max(1, n_articles)
        total_tokens_gpt += crit_tokens_gpt
        total_tokens_deepseek += crit_tokens_deepseek
        model_prices = [
            {"Model": "gpt-3.5-turbo", "Price": 0.0005, "Tokens": total_tokens_gpt},
            {"Model": "gpt-4", "Price": 0.01, "Tokens": total_tokens_gpt},
            {"Model": "deepseek-chat", "Price": 0.0002, "Tokens": total_tokens_deepseek},
            {"Model": "deepseek-coder", "Price": 0.0002, "Tokens": total_tokens_deepseek}
        ]
        price_df = pd.DataFrame(model_prices)
        price_df['Estimated Cost (USD)'] = 2 * price_df['Price'] * (price_df['Tokens'] / 1000)
        st.markdown('<h4 style="font-size:1.1em;font-weight:600;margin-top:0.5em;margin-bottom:0.5em;">Estimated Screening Cost by Model</h4>', unsafe_allow_html=True)
        chart = alt.Chart(price_df).mark_bar(size=30).encode(
            x=alt.X('Model:N', axis=alt.Axis(title='Model', labelAngle=0)),
            y=alt.Y('Estimated Cost (USD):Q', axis=alt.Axis(title='Estimated Cost (USD)', format="$,.2f")),
            tooltip=['Model', alt.Tooltip('Estimated Cost (USD):Q', format="$,.4f")]
        ).configure_view(
            fill='transparent'
        ).configure_axis(
            grid=False,
            domain=False,
            labelColor='#fff',
            titleColor='#fff',
            ticks=False,
            tickColor='transparent',
            tickBand='extent'
        ).configure_header(
            labelColor='#fff',
            titleColor='#fff',
            labelFontSize=13,
            titleFontSize=14,
            labelFontWeight='normal'
        ).configure(
            background='transparent'
        )
        st.altair_chart(chart, use_container_width=True)

        # --- F1-Score Comparison Graph (side-by-side bars) ---
        f1_data = [
            {"Model": "Human Alpha", "F1-Score": 0.819},
            {"Model": "Human Bravo", "F1-Score": 0.804},
            {"Model": "Human Charlie", "F1-Score": 0.832},
            {"Model": "GPT-3.5", "F1-Score": 0.628},
            {"Model": "GPT-4", "F1-Score": 0.732},
            {"Model": "GPT-4o", "F1-Score": 0.862},
            {"Model": "Gemini 1.5 Pro", "F1-Score": 0.813},
            {"Model": "LLaMA 3", "F1-Score": 0.695},
            {"Model": "Sonnet 3.5", "F1-Score": 0.869}
        ]
        f1_df = pd.DataFrame(f1_data)
        st.markdown('<h4 style="font-size:1.1em;font-weight:600;margin-top:1em;margin-bottom:0.5em;">F1-Score by Model (Literature Results)</h4>', unsafe_allow_html=True)
        f1_chart = alt.Chart(f1_df).mark_bar(size=30).encode(
            x=alt.X('Model:N', axis=alt.Axis(title='Model', labelAngle=0)),
            y=alt.Y('F1-Score:Q', axis=alt.Axis(title='F1-Score', format=".2f")),
            color=alt.Color('Model:N', scale=alt.Scale(range=['#0a84ff'])),
            tooltip=['Model', alt.Tooltip('F1-Score:Q', format=".3f")]
        ).configure_view(
            fill='transparent'
        ).configure_axis(
            grid=False,
            domain=False,
            labelColor='#fff',
            titleColor='#fff',
            ticks=False,
            tickColor='transparent',
            tickBand='extent'
        ).configure_header(
            labelColor='#fff',
            titleColor='#fff',
            labelFontSize=13,
            titleFontSize=14,
            labelFontWeight='normal'
        ).configure(
            background='transparent'
        )
        st.altair_chart(f1_chart, use_container_width=True)

        # --- Test on 100 Abstracts Option ---
        test_on_100 = st.checkbox("Test on 10 abstracts before full dataset (recommended)", value=False)
        st.caption("Test on a random batch of 10 abstracts and perform a human review to estimate the F1-score for your run before scaling to the full dataset.")

        if st.button("Start Screening", use_container_width=True):
            criteria = {
                "inclusion": inclusion_criteria.split('\n'),
                "exclusion": exclusion_criteria.split('\n')
            }
            
            # Create a placeholder for status text
            status_text = st.empty()
            
            # Start the screening process
            if test_on_100 and st.session_state.get('pubmed_df') is not None:
                df = st.session_state['pubmed_df']
                sample_df = df.sample(n=min(10, len(df)), random_state=42)
                sample_df.to_csv("pubmed_search_results.csv", index=False)
            
            # Start the screening process in a separate thread
            screening_thread = threading.Thread(
                target=lambda: run_screening(criteria)
            )
            screening_thread.start()
            
            # Update status while screening is running
            while screening_thread.is_alive():
                progress = get_screening_progress()
                current = progress['current']
                total = progress['total']
                if total > 0:
                    status_text.text(f"Processing paper {current} of {total}")
                time.sleep(0.1)  # Update every 100ms
            
            # After screening completes, run the subsequent scripts
            try:
                status_text.text("Combining results...")
                subprocess.run(['python', 'combine_results.py'], check=True)
                
                status_text.text("Converting to CSV...")
                subprocess.run(['python', 'json_to_csv.py'], check=True)
                
                status_text.text("Screening completed!")
                st.session_state.screening_completed = True
                st.success("Screening completed! Results are ready.")
                
                # Force a rerun to show the results
                st.experimental_rerun()
                
            except subprocess.CalledProcessError as e:
                st.error(f"Error in post-processing: {str(e)}")
                status_text.text("Error occurred during post-processing")

# Step 3: Results
if st.session_state.screening_completed:
    with st.container():
        st.header("📊 Step 3: Results")
        
        try:
            results_df = pd.read_csv("decision/Combined_Aging_Screening.csv")
            
            # Display summary
            st.subheader("Summary")
            total_papers = len(results_df)
            included_papers = len(results_df[results_df['decision'] == 'Include'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Papers", total_papers)
            with col2:
                st.metric("Included Papers", included_papers)
            with col3:
                st.metric("Excluded Papers", total_papers - included_papers)
            
            # Display results table
            st.subheader("Detailed Results")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="screening_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>Built by Malik Ahmed and Matthew Whitaker in the HDA-ML Lab at Imperial College London</p>
        <p>Last Updated: May 2025</p>
    </div>
    """,
    unsafe_allow_html=True
) 
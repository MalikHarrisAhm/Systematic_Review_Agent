# AI-Powered Systematic Review Assistant

An intelligent tool that automates the systematic review screening process using advanced AI models.

## Getting Started Guide

This tool helps you automate the process of screening research papers for systematic reviews. Here's how to use it:

1. **First Steps**
   - You'll need to obtain API keys for the services we use (instructions below)
   - The tool runs in your web browser, so no technical setup is required if you use the hosted version

2. **Using the Tool**
   - Choose how to search for papers:
     * **Manual Search**: Enter your PubMed search query using PubMed's advanced search syntax
     * **AI-Generated Search**: Describe your research topic in plain language, and our AI will generate appropriate search terms
   - The tool will search PubMed and retrieve relevant papers
   - It will analyze and categorize papers based on your criteria
   - You can review and adjust the AI's decisions
   - Export your results for further analysis

3. **Required API Keys**
   - **DeepSeek API Key** (Primary)
     1. Go to [DeepSeek AI](https://platform.deepseek.com/)
     2. Sign up for a free account
     3. Navigate to API Keys section
     4. Create a new API key
   
   - **OpenAI API Key** (Alternative)
     1. Visit [OpenAI Platform](https://platform.openai.com/)
     2. Sign up or log in
     3. Go to API Keys section
     4. Create a new secret key
   
   - **NCBI Entrez Credentials** (Required for PubMed search)
     1. Go to [NCBI Account](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
     2. Sign in or create an account
     3. Request an API key
     4. Use your email address as the Entrez email

## How It Works (Technical Overview)

The system uses a combination of AI technologies to automate systematic review screening:

1. **Paper Collection**
   - Searches PubMed using the NCBI Entrez API
   - Supports both manual and AI-generated search queries
   - Handles large result sets with automatic batching

2. **Screening Process**
   - Uses AI models to evaluate papers against your inclusion/exclusion criteria
   - Provides detailed explanations for each decision (include/exclude)
   - Processes papers in parallel for efficiency
   - Real-time progress tracking
   - Saves results in both JSON and CSV formats
   - Supports batch processing with progress tracking

3. **User Interface**
   - Modern web-based interface for easy interaction
   - Real-time progress tracking and visual feedback
   - Cost estimation for different AI models
   - Export results in CSV or JSON format

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
ENTREZ_EMAIL=your_email@example.com
ENTREZ_API_KEY=your_entrez_api_key
```

5. Run the application locally:
```bash
streamlit run app.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Add your secrets (API keys) in the "Secrets" section
7. Deploy!

### 2. Heroku

1. Create a `Procfile`:
```
web: streamlit run app.py
```

2. Create a `runtime.txt`:
```
python-3.9.18
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

4. Set environment variables:
```bash
heroku config:set OPENAI_API_KEY=your_key
heroku config:set DEEPSEEK_API_KEY=your_key
heroku config:set ENTREZ_EMAIL=your_email
heroku config:set ENTREZ_API_KEY=your_key
```

### 3. Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t systematic-review-agent .
docker run -p 8501:8501 systematic-review-agent
```

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `ENTREZ_EMAIL`: Your email for NCBI Entrez
- `ENTREZ_API_KEY`: Your NCBI Entrez API key

## License

MIT License
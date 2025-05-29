# AI-Powered Systematic Review Assistant

An intelligent tool that automates the systematic review screening process using advanced AI models.

## Instructions for Non-ML Specialists

This tool helps you automate the process of screening research papers for systematic reviews. Here's how to use it:

1. **Getting Started**
   - You'll need to obtain API keys for the services we use (instructions below)
   - The tool runs in your web browser, so no technical setup is required if you use the hosted version

2. **Using the Tool**
   - Enter your research question or topic
   - The tool will automatically search relevant databases for papers
   - It will analyze and categorize papers based on your criteria
   - You can review and adjust the AI's decisions
   - Export your results for further analysis

3. **Required API Keys**
   - OpenAI API key (for paper analysis)
   - DeepSeek API key (for additional analysis)
   - NCBI Entrez credentials (for paper search)
   - Don't worry if you don't have these - we can help you obtain them

## How It Works (Technical Overview)

The system uses a combination of AI technologies to automate systematic review screening:

1. **Paper Collection**
   - Searches multiple academic databases using the NCBI Entrez API
   - Automatically retrieves full text when available
   - Handles different paper formats and sources

2. **AI Analysis**
   - Uses OpenAI's GPT models to understand paper content
   - Employs DeepSeek for additional analysis and verification
   - Identifies key information like study design, population, and outcomes

3. **Screening Process**
   - Automatically categorizes papers based on inclusion/exclusion criteria
   - Provides confidence scores for its decisions
   - Allows human review and correction of AI decisions

4. **User Interface**
   - Modern web-based interface for easy interaction
   - Real-time progress tracking and visual feedback
   - Simple export functionality for results

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
# Systematic Review AI Agent

A Streamlit application for automated systematic review screening using AI.

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
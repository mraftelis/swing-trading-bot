# Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["bash", "-c", "python main_pipeline.py && streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]

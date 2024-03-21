FROM python:3.9-slim

EXPOSE 8501

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD streamlit run /app/src/dashboard.py --server.port=8501 --server.address=0.0.0.0

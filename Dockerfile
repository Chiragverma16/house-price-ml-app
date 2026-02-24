FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501



CMD sh -c "uvicorn backend.fast_app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"
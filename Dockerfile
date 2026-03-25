FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Add --upgrade and remove strict hash checking
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python embed.py

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
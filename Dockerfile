FROM python:3.10.6

# Set working directory inside the container
# WORKDIR /app

# Copy the entire project into the container
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY /model_registry /model_registry

COPY /spark /spark
# Run the FastAPI app
CMD uvicorn spark.api.fast:app --host 0.0.0.0 --port $PORT

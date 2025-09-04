FROM python:3.10.6-buster

# Set working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run the FastAPI app
CMD uvicorn spark.api.fast:app --host 0.0.0.0 --port $PORT

# Use Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app
COPY sentiment_model.pkl /app

# Install dependencies
RUN pip install flask pandas scikit-learn

# Expose port
EXPOSE 8080

# Command to run the app
CMD ["python", "app.py"]

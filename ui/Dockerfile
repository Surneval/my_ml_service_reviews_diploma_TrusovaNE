# ui/Dockerfile


FROM python:3.11-slim


# environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=600

# work directory
WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# install dependencies with enhanced pip settings
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple/ --retries 5 --timeout 200 --prefer-binary --progress-bar off

# create a non-root user with a valid home directory
RUN addgroup --system appgroup && \
    adduser --ingroup appgroup --home /home/appuser --disabled-password --gecos "" appuser

# assign ownership of the work directory to the non-root user
RUN mkdir -p /app && chown -R appuser:appgroup /app

# switch to the non-root user
USER appuser

# copy application code
COPY app.py .

# expose the port Streamlit is running on
EXPOSE 8501

# run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Acko AI Platform - the web app as a Docker image (runs on the AWS EC2 server, port 8501).
#
#   docker build -t acko .
#   docker run -d --name acko -p 8501:8501 --env-file acko.env acko
#
# The settings and secrets (database, bucket, Gemini key, manager password) are NOT in the image: they come in as
# environment variables when the container starts (--env-file). deploy/ec2_deploy.sh does all of this on the server.

FROM python:3.11-slim

# MPLBACKEND=Agg: charts are drawn without a screen. The file watcher is off because nobody edits files in a running container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# libgomp1 is the OpenMP library that numba (used by SHAP) can load. Everything else the packages need comes inside their wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# The app runs as a normal user, not as root.
RUN useradd --create-home --uid 1000 acko
WORKDIR /app
RUN chown acko:acko /app

# 1. Packages first: this layer is rebuilt only when requirements.txt or constraints.txt change.
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# 2. The app itself (what .dockerignore leaves in: code, models/, data/policy_pdfs/, .streamlit/).
USER acko
COPY --chown=acko:acko . .

# 3. The vector store: read the policy PDFs and fill ChromaDB (also downloads the small embedding model once, about 80 MB).
#    It is built here, as the same user that runs the app, so the container never has to download or build anything on start.
RUN python -m scripts.build_index

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

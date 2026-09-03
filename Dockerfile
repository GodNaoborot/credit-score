FROM python:3.12-slim

# Hugging Face Spaces runs the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user src/ ./src/
COPY --chown=user app/ ./app/
COPY --chown=user artifacts/scorecard.json ./artifacts/

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

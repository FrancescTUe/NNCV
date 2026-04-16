FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


# install dependencies (add any additional dependencies your project requires)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    pillow \
    numpy \
    opencv-python \
    six \
    wandb \
    torchmetrics \
    ptflops \
    transformers

WORKDIR /app
COPY model.py /app/model.py
#COPY model.py /app/model_entropy.py
COPY predict_ood.py /app/predict.py
COPY teacher_model.pt "/app/model.pt"
COPY ood_model.pt "/app/ood_model.pt"
COPY segformer /app/segformer


ENTRYPOINT ["python", "/app/predict.py"]
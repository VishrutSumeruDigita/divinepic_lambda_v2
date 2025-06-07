FROM public.ecr.aws/lambda/python:3.10-x86_64

# Install system dependencies
RUN yum update -y && \
    yum install -y wget unzip gcc gcc-c++ cmake make && \
    yum clean all

# Set environment variables for Lambda
ENV MPLCONFIGDIR=/app/.matplotlib
ENV NUMBA_CACHE_DIR=/tmp
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV INSIGHTFACE_ROOT=/app/.insightface

# Download and setup AntelopeV2 model files
RUN cd /tmp && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip && \
    if [ ! -f "antelopev2.zip" ]; then echo "Failed to download model"; exit 1; fi && \
    unzip -q antelopev2.zip -d antelopev2_files && \
    if [ ! -d "antelopev2_files" ]; then echo "Failed to extract model"; exit 1; fi && \
    mkdir -p /app/models/antelopev2/detection && \
    cp antelopev2_files/antelopev2/scrfd_10g_bnkps.onnx /app/models/antelopev2/detection/ && \
    cp antelopev2_files/antelopev2/glintr100.onnx /app/models/antelopev2/ && \
    if [ ! -f "/app/models/antelopev2/detection/scrfd_10g_bnkps.onnx" ] || [ ! -f "/app/models/antelopev2/glintr100.onnx" ]; then \
        echo "Failed to copy model files"; exit 1; \
    fi && \
    rm -rf /tmp/antelopev2*

# Create necessary directories
RUN mkdir -p /tmp/uploads && \
    mkdir -p /app/.insightface && \
    mkdir -p /app/.matplotlib

# Copy function code
ADD serving_api.tar.gz ${LAMBDA_TASK_ROOT}

# Install Python packages with specific strategies for ML libraries
# Install NumPy first to prevent version conflicts
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} "numpy>=1.24.3,<2.0" && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} fastapi==0.104.1 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} mangum==0.17.0 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} python-dotenv==1.0.1 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} boto3==1.24.53 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} python-multipart==0.0.6 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} pillow==10.0.1 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} tqdm==4.66.1 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} elasticsearch==8.10.0 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} transformers==4.36.0 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} --find-links https://download.pytorch.org/whl/torch_stable.html torch==2.1.0+cpu && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} --find-links https://download.pytorch.org/whl/torch_stable.html torchvision==0.16.0+cpu && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} opencv-python-headless==4.8.1.78 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} onnxruntime==1.16.3 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} onnx==1.15.0 && \
    pip3 install --no-cache-dir --target ${LAMBDA_TASK_ROOT} insightface==0.7.3

CMD [ "serving_api.lambda_handler" ]
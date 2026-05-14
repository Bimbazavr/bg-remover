FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pillow python-multipart httpx \
    "numpy<3" onnxruntime scikit-image scipy tqdm pooch requests jsonschema && \
    pip install --no-cache-dir --no-deps rembg

# Patch rembg to skip pymatting
RUN python -c "\
import re, pathlib; \
p = pathlib.Path('/usr/local/lib/python3.11/site-packages/rembg/bg.py'); \
t = p.read_text(); \
t = t.replace(\
'from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf\nfrom pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml\nfrom pymatting.util.util import stack_images',\
'try:\n    from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf\n    from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml\n    from pymatting.util.util import stack_images\nexcept ImportError:\n    estimate_alpha_cf = estimate_foreground_ml = stack_images = None'\
); p.write_text(t); print('patched')"

# Pre-download u2net model
RUN python -c "from rembg import new_session; new_session('u2net')"

COPY . .

RUN mkdir -p backgrounds

EXPOSE 7860
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}

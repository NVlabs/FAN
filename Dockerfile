FROM nvcr.io/nvidia/pytorch:22.06-py3

COPY requirements.txt .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY . /workdir/FAN/
WORKDIR /workdir/FAN

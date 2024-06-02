FROM nvcr.io/nvidia/pytorch:24.03-py3
WORKDIR /root
COPY requirements.txt ./
RUN pip install -r requirements.txt --upgrade
COPY app/ ./
ENTRYPOINT [ "python3", "app.py" ]

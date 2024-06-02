#!/env/bin/bash
docker run -d --rm --name ai4se-api-test-1 --gpus "device=1" -v /var/opt/huggingface:/root/.cache/huggingface -p0.0.0.0:8010:8000 ai4se-1-api
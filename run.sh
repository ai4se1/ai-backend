#!/bin/env bash
docker run -d --rm --name ai4se-1-api --gpus "device=1" -v /var/opt/huggingface:/root/.cache/huggingface -p0.0.0.0:8010:8000 ghcr.io/ai4se1/ai-backend:main

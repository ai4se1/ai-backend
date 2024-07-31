# AI-Backend

## WARNING

This is only a prototype. The API has no means of authentication or authorization. If you plan on deploying this service on an internet facing server, please use a reverse proxy with basic authentication or another form of authentication to secure your server!

## Description

The ai-backend is the backend for the llm-debug VS Code extension. The backend is responsible for generating annotations, that can help developers to debug their code. The backend runs in a docker container. A new image gets automatically build and published to the GitHub docker registry for every commit. To run the latest prebuilt container use the script in `run.sh` or modify it to your needs. You can download and execute the script with the following one-liner: `curl https://raw.githubusercontent.com/ai4se1/ai-backend/main/run.sh | bash`. Please make sure to review the contents of the script before executing it on your machine!!

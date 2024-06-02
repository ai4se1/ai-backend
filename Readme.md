# Readme

To run the container on the server run `bash run.sh`

To rebuild the container after making changes to `app.py` copy the files using `scp -r . ai4se:` and run `docker build . -t ai4se-1-api` in your home directory on the server. You might have to replace ai4se with the appropriate ssh host.

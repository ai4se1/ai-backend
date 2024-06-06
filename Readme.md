# Readme

To run the container on the server run `bash run.sh`

To rebuild the container after making changes to `app.py` copy the files using `scp -r . ai4se:` and run `docker build . -t ai4se-1-api` in your home directory on the server. You might have to replace ai4se with the appropriate ssh host.

To use the copy command add the following lines to `~/.ssh/config`

```text
Host ai4se
    HostName delos.eaalab.hpi.uni-potsdam.de
    User Alexander.Sohn
    Port 22
    IdentityFile ~/.ssh/ai4se
```

and replace the values accordingly

## Current problems and ideas

- multiline responses are not properly working
- Maybe use regexes to get better matching than python find
- detect when response is incomplete (e.g. no <eos> token)
- Better fallback for not unique -> Just output all
- Only regenerate one item in the recursion not all

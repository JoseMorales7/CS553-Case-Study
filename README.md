# Canvas Critic

Canvas Critic is a Gradio app that scores an uploaded artwork and suggests
improvements. It runs as a normal local web server and listens on port 8000.

## Run locally

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then start the app:

```powershell
python app.py
```

Open `http://localhost:8000` on the host computer. Other people on the same
network can use `http://<host-ip-address>:8000`. The operating-system firewall
must allow inbound TCP traffic on port 8000.

Each visitor can enter their own Hugging Face token in the masked token field.
That token is used for that visitor's hosted inference request and is not read
from the server's Hugging Face login or written to disk by the app. Because the
default local connection uses plain HTTP, only enter credentials on a trusted
network. Use an HTTPS reverse proxy before exposing the app over the internet.
Visitors who do not provide a token can select the local model instead.

The default bind address and port can be overridden when needed:

```powershell
$env:GRADIO_SERVER_NAME = "127.0.0.1"
$env:GRADIO_SERVER_PORT = "9000"
python app.py
```

Selecting the local model downloads and loads the approximately 16 GB ArtiMuse
checkpoint. Set `PRELOAD_LOCAL_MODEL=1` to load it at startup; otherwise it is
loaded the first time it is used.

<!-- # This file was creating with GPT6.1 sol in response to the following query

# Could you now modify the docker/compose.yaml file such that it spins up the app container with some image name (i'll replace later), 
# the container for the prometheus collection server, the container for the grafana server, and the container for ngrok? -->

# Docker deployment

The Compose stack runs the app, Prometheus, Grafana, and ngrok on one shared
bridge network named `monitoring`. It uses official service images for the monitoring and tunnel
services; their Dockerfiles are not needed by this Compose configuration.

1. Replace `cs553-app:latest` in `compose.yaml`, or set `APP_IMAGE` to your app
   image name. The image must run `python app.py` to start Gradio on port 7860
   and the metrics exporter on port 8000. The existing node-exporter scrape job
   also expects the image to start node exporter on port 9100.
2. Copy `.env.example` to `.env` in this directory and fill in the token and
   Grafana password. `.env` is ignored by Git. Alternatively, keep an environment
   file on the server and pass its absolute path with `--env-file`.
3. From the repository root, start the stack:

   ```sh
   docker compose -f docker/compose.yaml up -d
   ```

   For an environment file stored elsewhere on a Linux server:

   ```sh
   docker compose --env-file /etc/cs553/secrets.env -f docker/compose.yaml up -d
   ```

| Service | Address |
| --- | --- |
| Gradio app | `http://localhost:7860` or `http://server-address:7860` |
| Grafana | `http://localhost:3000` or `http://server-address:3000` |
| Prometheus | `http://localhost:9090` on the Docker host (loopback only) |
| Public app | ngrok URL shown by `docker compose -f docker/compose.yaml logs ngrok` |

Log in to Grafana using `GRAFANA_ADMIN_USER` (default `admin`) and the configured
password. Its default Prometheus data source is provisioned automatically at
`http://prometheus:9090`; dashboards can be created in Grafana's UI. The admin
credentials initialize a new Grafana database; changing the environment file
does not reset an existing user's password.

Prometheus scrapes `cs553-app:8000/metrics` and `cs553-app:9100/metrics` every
15 seconds. Metrics ports do not need host publishing. If your replacement
image does not include node exporter, remove its scrape job. Only the Gradio
port is forwarded by ngrok. Service dependencies establish startup order;
scraping and the tunnel may report connection errors briefly while the app
initializes.

Prometheus history and Grafana configuration persist in named Docker volumes
when containers are recreated. `docker compose -f docker/compose.yaml down`
preserves them; adding `--volumes` removes their data. The images use `latest`
tags for initial setup; pin tested versions or digests for repeatable deployment.

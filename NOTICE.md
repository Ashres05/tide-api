# tide-api is now managed by systemd

As of 2026-04-27 the API is run via the systemd service `tide-api.service`,
not manually with `python -m uvicorn`. The service auto-starts on boot,
restarts on failure, and binds port 8000 — so a manual `uvicorn` run will
fail with "address already in use".

## Lifecycle

    sudo systemctl restart tide-api     # restart after `git pull`
    sudo systemctl status  tide-api     # check status
    sudo systemctl stop    tide-api     # stop (e.g. to run manually)
    sudo systemctl start   tide-api

## Live logs (replaces the old "watch foreground uvicorn" experience)

    sudo journalctl -u tide-api -f                 # tail forever
    sudo journalctl -u tide-api --since "5 min ago"
    sudo journalctl -u tide-api -n 200             # last 200 lines
    sudo journalctl -u tide-api -p err             # errors only
    sudo journalctl -u tide-api --since today

Logs persist across reboots (journald Storage=persistent).
PYTHONUNBUFFERED=1 is set, so output appears immediately.

## Cloudflare Tunnel (also systemd)

    sudo systemctl status cloudflared
    sudo journalctl -u cloudflared -f

## Files

- Unit:           /etc/systemd/system/tide-api.service
- Wrapper:        /home/ubuntu/tide-data-pipeline/tide-api/run_api.sh
- Env (TIDE):     /home/ubuntu/tide-data-pipeline/tide-api/.env
- Env (Snowflake):/home/ubuntu/tide-data-pipeline/tide-api/secrets/amg_research.env

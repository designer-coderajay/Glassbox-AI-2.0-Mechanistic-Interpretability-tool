"""
Serverless proxy for pypistats download counts.
pypistats.org does not send CORS headers, so browser JS cannot fetch it directly.
This function runs server-side on Vercel, fetches the data, and re-serves it
with proper CORS + cache headers.
"""
from http.server import BaseHTTPRequestHandler
import json
import urllib.request


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            url = "https://pypistats.org/api/packages/glassbox-mech-interp/recent"
            req = urllib.request.Request(url, headers={"User-Agent": "glassbox-site/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read())
            data = payload.get("data", {})
            body = json.dumps({
                "last_day":   data.get("last_day", 0),
                "last_week":  data.get("last_week", 0),
                "last_month": data.get("last_month", 0),
            }).encode()
            self._respond(200, body)
        except Exception:
            # Return zeros — the JS will keep its hardcoded fallback.
            body = json.dumps({"last_day": 0, "last_week": 0, "last_month": 0}).encode()
            self._respond(200, body)

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        # Cache on the CDN edge for 1 hour, accept stale for 24 h.
        self.send_header("Cache-Control", "s-maxage=3600, stale-while-revalidate=86400")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence default stderr logging
        pass

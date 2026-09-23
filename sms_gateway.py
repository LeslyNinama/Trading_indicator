#!/usr/bin/env python3
"""
SMS gateway for YOUR Android phone - no SMS provider, no API account. Uses your own SIM.

Run this ON THE PHONE (Termux). The trading monitor on your PC posts alerts to it over Wi-Fi,
and the phone sends them as normal SMS.

ONE-TIME PHONE SETUP
  1. Install Termux and Termux:API from F-Droid (the Play Store builds are outdated).
  2. In Termux:   pkg install python termux-api
  3. Android Settings > Apps > Termux:API > Permissions > allow SMS (and Phone).
  4. Copy this file to the phone (or:  pkg install curl  and download it), then run:
         python sms_gateway.py --token MY_SECRET
  5. Find the phone's Wi-Fi IP (Settings > Wi-Fi > your network). Keep the phone and PC on the
     same network (a phone hotspot works too) and keep Termux running (acquire wake lock:
     pull down the Termux notification > "Acquire wakelock").

IN THE APP:  provider "Phone (Termux gateway)", URL  http://<phone-ip>:8765 , token MY_SECRET.

Only requests carrying the correct token are accepted. Use it on a network you trust.
Note: SIM plans usually cap SMS per day (about 100 in India); each alert goes to every number.
"""
import argparse
import hmac
import json
import re
import secrets
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = ""


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ping":
            self._reply(200, {"ok": True, "termux_sms": bool(shutil.which("termux-sms-send"))})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/send":
            return self._reply(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 10_000:
            return self._reply(413, {"error": "bad size"})
        try:
            data = json.loads(self.rfile.read(n))
        except ValueError:
            return self._reply(400, {"error": "bad json"})
        if not hmac.compare_digest(str(data.get("token", "")).encode(), TOKEN.encode()):
            return self._reply(403, {"error": "bad token"})
        numbers = [str(x) for x in data.get("numbers", []) if re.fullmatch(r"\d{10}", str(x))][:20]
        message = str(data.get("message", ""))[:300]
        if not numbers or not message:
            return self._reply(400, {"error": "need numbers and message"})
        results = []
        for num in numbers:                       # one SMS per recipient
            try:
                p = subprocess.run(["termux-sms-send", "-n", "+91" + num, message],
                                   capture_output=True, text=True, timeout=30)
                ok, detail = p.returncode == 0, (p.stderr or "sent").strip()[:80]
            except Exception as e:
                ok, detail = False, str(e)[:80]
            results.append({"number": num, "ok": ok, "detail": detail})
            print(f"SMS to {num}: {'OK' if ok else 'FAILED - ' + detail}", flush=True)
        self._reply(200, {"results": results})

    def log_message(self, *a):                    # keep the console quiet
        pass


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="shared secret (random one is generated if omitted)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    TOKEN = a.token or secrets.token_urlsafe(9)
    if not shutil.which("termux-sms-send"):
        print("WARNING: termux-sms-send not found. Run:  pkg install termux-api")
    print(f"SMS gateway on port {a.port}\nToken: {TOKEN}\nCtrl+C to stop.")
    HTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

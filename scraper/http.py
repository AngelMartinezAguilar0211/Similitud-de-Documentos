# scraper/http.py
import os
import time
import logging
import requests

log = logging.getLogger(__name__)

class HttpClient:
    def __init__(self, sleep: float = 0.5, timeout: int = 20):
        self.session = requests.Session()
        self.sleep = sleep
        self.timeout = timeout
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; NLP-Practica2/1.0; +https://github.com/yourproject)"
        })
        cookie_str = os.environ.get("ARXIV_COOKIE")
        if cookie_str:
            jar = requests.cookies.RequestsCookieJar()
            for part in cookie_str.split(";"):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    jar.set(k, v)
            self.session.cookies.update(jar)

    def get(self, url: str, params=None, raise_for_status: bool = True):
        tries = 3
        for i in range(tries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if raise_for_status:
                    r.raise_for_status()
                return r
            except Exception as e:
                log.warning("[HTTP] intento %d falló para %s: %r", i+1, url, e)
                time.sleep(self.sleep * (i+1))
        r = self.session.get(url, params=params, timeout=self.timeout)
        if raise_for_status:
            r.raise_for_status()
        return r

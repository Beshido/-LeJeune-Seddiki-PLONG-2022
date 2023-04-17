import http.server
import logging
logging.basicConfig(level = logging.INFO)
import os
import socketserver
import urllib.parse

HOST = "0.0.0.0"
DIRECTORY = "html/build/dist/"

class MyServer(http.server.BaseHTTPRequestHandler):
    def __init__(self, request: bytes, client_address: tuple, server: socketserver.BaseServer) -> None:
        super().__init__(request, client_address, server)

    def do_GET(self) -> None:
        request = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(request.query)
        path = request.path if request.path != '/' else "/index.html"

        self.send_response(200)
        if path.endswith(".html"):
            self.send_header("Content-type", "text/html")
        if path.endswith(".css"):
            self.send_header("Content-type", "text/css")
        elif path.endswith(".js"):
            self.send_header("Content-type", "text/javascript")
        elif path.endswith(".png"):
            self.send_header("Content-type", "image/png")
        self.end_headers()
                    
        try:
            with open(path[1:], 'rb') as file:
                self.wfile.write(file.read())
        except FileNotFoundError:
            print(f"Requested file does not exist: {self.path}")

    def do_POST(self) -> None:
        request = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(request.query)
        path = request.path if request.path != '/' else "/index.html"
        data = self.rfile.read(int(self.headers["Content-Length"]))
        logging.info(data)

        self.send_response(200)
        self.end_headers()


def start(port: int = 8080) -> None:
    webServer = http.server.HTTPServer((HOST, port), MyServer)
    logging.info(f"Le serveur a été lancé à l'adresse suivante : http://{HOST}:{port}")

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.server_close()
    logging.info("Le serveur a été arrêté.")
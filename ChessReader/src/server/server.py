import http.server
import logging
logging.basicConfig(level = logging.INFO)
import socket
import socketserver
import urllib.parse
from src.board_reader import model

HOST = "0.0.0.0"

class MyServer(http.server.BaseHTTPRequestHandler):
    def __init__(self, request: tuple, client_address: tuple, server: socketserver.BaseServer) -> None:
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
        self.send_response(200)
        self.end_headers()

        logging.info(f"Image reçue, début de la prédiction...")
        try:
            data = model.predict_from_memory(data)
            logging.info(f"Prédiction réussie.")
            self.wfile.write(b"OK")
            self.wfile.write(data)
        except ValueError:
            logging.info(f"Échec de la prédiction.")
            self.wfile.write(b"NO")


def start(port: int = 8080) -> None:
    webServer = http.server.HTTPServer((HOST, port), MyServer)
    logging.info(f"Le serveur a été lancé et est accessible via l'adresse {socket.gethostbyname(socket.gethostname())} au port {port}. Pour arrêter le serveur, appuyez sur Ctrl+C.")

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.server_close()
    logging.info("Le serveur a été arrêté.")
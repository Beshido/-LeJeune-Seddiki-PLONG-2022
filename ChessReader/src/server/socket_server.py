import logging
logging.basicConfig(level = logging.INFO)
import socket
from src.board_reader import model
from src.ia import best_move

HOST = "0.0.0.0"
"""Adresse IP du serveur socket."""

SIZE_OF_HEADER = 5
"""Taille du header envoyé par les applications dans le protocole de communication."""

PHOTO_HEADER = "PHOTO"
"""Header identifiant envoyé par l'application photo."""

VIBRATOR_HEADER = "VIBRA"
"""Header identifiant envoyé par l'application vibreuse."""

SIZE_OF_INT = 4
"""Taille d'un entier en octets."""

def _launch(server_socket: socket.socket) -> None:
    """Boucle du serveur socket. Attends la connexion des deux applications, puis loop sur leur back and forth tant qu'elles sont connectées."""

    logging.info("Attente de la connexion des deux applications...")
    photo_taker_socket, vibrator_socket = None, None
    while photo_taker_socket is None or vibrator_socket is None:
        current_socket, _ = server_socket.accept()
        message = current_socket.recv(SIZE_OF_HEADER).decode("utf-8")
        if message == PHOTO_HEADER:
            photo_taker_socket = current_socket
            logging.info("Application photo connectée.")
        elif message == VIBRATOR_HEADER:
            vibrator_socket = current_socket
            logging.info("Application vibreuse connectée.")
        else:
            logging.info("Message inconnu reçu: " + message)

    logging.info("Les deux applications sont connectées.")
    while True:
        logging.info("Attente de l'envoi d'une photo...")
        image_size = int.from_bytes(photo_taker_socket.recv(SIZE_OF_INT), "little", signed=True)
        if (image_size <= 0):
            logging.info("Taille de l'image invalide. Recherche de nouveaux candidats.")
            break
        logging.info(f"Préparation à la réception d'une image de {image_size} octets...")
        image = photo_taker_socket.recv(image_size, socket.MSG_WAITALL)
        logging.info(f"Image de {len(image)} octets reçue, début de la prédiction...")
        try:
            data = model.predict_from_memory(image)
            move = best_move.get_best_move(data)
            if move is None:
                logging.info("Échec lors de la sélection du meilleur move via StockFish. Envoi d'un message d'erreur au vibreur...")
                vibrator_socket.send(b"ERROR")
            else:
                logging.info(f"Prediction terminée. Meilleur move sélectionné: {move}. Envoi au vibrateur...")
                vibrator_socket.send(move)
            logging.info("Données envoyées au vibrateur.")

        except ValueError as e:
            logging.info(e)
            logging.info("Échec de la prédiction. Envoi d'un message d'erreur à l'appareil photo...")
            photo_taker_socket.send(b"FIXME")
            logging.info("Message d'erreur envoyé à l'appareil photo.")
    photo_taker_socket.close()
    vibrator_socket.close()
    logging.info("Socket photo et vibreur fermés.")

def start(port: int = 8080):
    """Lance le serveur socket au port indiqué."""

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, port))
    server_socket.listen(2)
    try:
        while True:
            _launch(server_socket)
    except KeyboardInterrupt:
        logging.info("Fermeture du socket.")
    server_socket.close()
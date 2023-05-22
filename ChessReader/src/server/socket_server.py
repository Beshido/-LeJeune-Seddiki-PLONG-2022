import logging
logging.basicConfig(level = logging.INFO)
import socket
import time
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

PICTURE_HEADER = "PICTU"
"""Header identifiant envoyé par l'application photo pour indiquer qu'elle envoie une photo."""

SEND_FEN_HEADER = "BOFEN"
"""Header identifiant envoyé par le serveur pour indiquer qu'elle envoie le FEN."""

RECEIVE_FIXED_FEN_HEADER = "FENSI"
"""Header identifiant envoyé par l'application photo pour indiquer qu'elle envoie le FEN corrigé par l'utilisateur."""

SEND_VIBRATOR_FEN_HEADER = "FENVI"
"""Header identifiant envoyé par le serveur pour indiquer qu'elle envoie le meilleur move au vibreur."""

SIZE_OF_INT = 4
"""Taille d'un entier en octets."""

def _launch(server_socket: socket.socket) -> None:
    """Boucle du serveur socket. Attends la connexion des deux applications, puis loop sur leur back and forth tant qu'elles sont connectées."""

    logging.info("Attente de la connexion des deux applications...")
    photo_taker_socket, vibrator_socket = None, None
    while photo_taker_socket is None or vibrator_socket is None:
        current_socket, _ = server_socket.accept()
        message = current_socket.recv(SIZE_OF_HEADER).decode()
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
        logging.info("Attente de l'envoi d'une photo ou d'un FEN...")
        message = photo_taker_socket.recv(SIZE_OF_HEADER).decode()
        if message == PICTURE_HEADER:
            image_size = int.from_bytes(photo_taker_socket.recv(SIZE_OF_INT), "little", signed=True)
            if (image_size <= 0):
                logging.info("Taille de l'image invalide. Recherche de nouveaux candidats.")
                break
            logging.info(f"Préparation à la réception d'une image de {image_size} octets...")
            image = photo_taker_socket.recv(image_size, socket.MSG_WAITALL)
            logging.info(f"Image de {len(image)} octets reçue, début de la prédiction...")
            try:
                data = model.predict_from_memory(image)
            except ValueError as e:
                logging.info(e)
                logging.info("Échec de la prédiction. Rententative...")
                continue
            
            fen = data.board_fen()
            logging.info("Envoi du FEN à corriger à l'appareil photo...")
            photo_taker_socket.send(SEND_FEN_HEADER.encode())
            photo_taker_socket.send(len(fen).to_bytes(SIZE_OF_INT, "little", signed=True))
            photo_taker_socket.send(fen.encode())
    
        elif message == RECEIVE_FIXED_FEN_HEADER:
            logging.info("Reception du FEN corrigé de l'appareil photo...")
            fen_size = int.from_bytes(photo_taker_socket.recv(SIZE_OF_INT), "little", signed=True)
            fen = photo_taker_socket.recv(fen_size, socket.MSG_WAITALL).decode()
            logging.info(f"FEN corrigé reçu : {fen}")
            move = best_move.get_best_move_from_fen(fen)
            if move is not None:
                logging.info(f"Prediction terminée. Meilleur move sélectionné: {move}. Envoi au vibrateur...")
                move_size = len(move.uci())
                vibrator_socket.send(SEND_VIBRATOR_FEN_HEADER.encode())
                vibrator_socket.send(move_size.to_bytes(SIZE_OF_INT, "little", signed=True))
                vibrator_socket.send(move.uci().encode())
                logging.info("Données envoyées au vibrateur.")
            else:
                logging.info("Échec lors de la sélection du meilleur move via StockFish. Rentattive...")
                continue
        else:
            logging.info("Header inconnu reçu. Retentative...")
            break

    photo_taker_socket.close()
    vibrator_socket.close()
    logging.info("Socket photo et vibreur fermés.")

def _test_server(server_socket: socket.socket) -> None:
    """Boucle de serveur rapide pour tester rapidement le prétraitement des images."""

    logging.info("Attente de la connexion...")
    photo_taker_socket = None
    while photo_taker_socket is None:
        current_socket, _ = server_socket.accept()
        message = current_socket.recv(SIZE_OF_HEADER).decode()
        if message == PHOTO_HEADER:
            photo_taker_socket = current_socket
            logging.info("Application photo connectée.")
        else:
            logging.info("Message inconnu reçu: " + message)

    while True:
        logging.info("Attente de l'envoi d'une photo...")
        message = photo_taker_socket.recv(SIZE_OF_HEADER).decode()
        if message != PICTURE_HEADER:
            logging.info(f"Header inconnu reçu : {message}. Retentative...")
            continue
        image_size = int.from_bytes(photo_taker_socket.recv(SIZE_OF_INT), "little", signed=True)
        if (image_size <= 0):
            logging.info("Taille de l'image invalide. Recherche de nouveaux candidats.")
            break
        logging.info(f"Préparation à la réception d'une image de {image_size} octets...")
        image = photo_taker_socket.recv(image_size, socket.MSG_WAITALL)
        logging.info(f"Image de {len(image)} octets reçue, début de la prédiction...")
        try:
            data = model.predict_from_memory(image)
            logging.info(f"Prédiction réussie : {data}")
        except ValueError as e:
            logging.info(e)
            logging.info("Échec de la prédiction. Rententative...")
            continue
        
    photo_taker_socket.close()
    logging.info("Socket photo fermé.")

def start(port: int = 8080):
    """Lance le serveur socket au port indiqué."""

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, port))
    server_socket.listen(2)
    try:
        while True:
            _launch(server_socket)
            # _test_server(server_socket)
    except KeyboardInterrupt:
        logging.info("Fermeture du socket.")
    server_socket.close()
import logging
logging.basicConfig(level = logging.INFO)
import socket
from src.board_reader import model

HOST = "0.0.0.0"

SIZE_OF_HEADER = 5
PHOTO_HEADER = "PHOTO"
VIBRATOR_HEADER = "VIBRA"

SIZE_OF_INT = 4

def start(port: int = 8080) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, port))
        s.listen(2)
        photo_taker_socket, vibrator_socket = None, None
        logging.info("Attente de la connexion des deux applications...")
        while photo_taker_socket is None or vibrator_socket is None:
            current_socket, _ = s.accept()
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
            logging.info(f"Préparation à la réception d'une image de {image_size} octets...")
            image = photo_taker_socket.recv(image_size)
            logging.info(f"Image de {image.__sizeof__()} octets reçue, début de la prédiction...")
            data = model.predict_from_memory(image)
            logging.info("Prédiction terminée, envoi des données au vibreur...")
            vibrator_socket.send(data.__sizeof__())
            vibrator_socket.send(data)
            logging.info("Données envoyées.")
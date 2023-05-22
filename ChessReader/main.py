#!/usr/bin/env python3

import argparse
import logging
logging.basicConfig(level = logging.INFO)
import pathlib
import pprint
import src.board_reader.model as model
import src.ia.best_move as best_move
import src.server.server as server
import src.server.socket_server as socket_server
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prépare l'image d'échiquier à être entraînée.")
    parser.add_argument("--predict", type=pathlib.Path, help="Chemin vers l'image à être analysée.")
    parser.add_argument("--build", action="store_true", help="Entraîne le modèle avec les images de jeu dans 'neural-network-dataset'.")
    parser.add_argument("--train", type=int, help="Entraine le dataset le nombre de fois indiqué.")
    parser.add_argument("--server", type=int, help="Lance le serveur au port indiqué.")
    parser.add_argument("--socket", type=int, help="Lance le serveur socket au port indiqué.")
    parser.add_argument("--preprocess", type=pathlib.Path, help="Affiche le résultat du pré-traitements de l'image indiquée.")
    parser.add_argument("--fen", type=pathlib.Path, help="Affiche le FEN d'une image d'échiquier via machine learning.")
    parser.add_argument("--bestmove", type=str, help="Affiche le meilleur move d'une image d'échiquier via Stockfish.")
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    if args.predict is not None:
        model.predict_from_file(args.predict)
    elif args.build:
        model.build_dataset_tree_structure()
    elif args.train is not None:
        model.train_model(args.train)
    elif args.server is not None:
        server.start(args.server)
    elif args.socket is not None:
        socket_server.start(args.socket)
    elif args.preprocess is not None:
        try: 
            data = model.preprocess.preprocess_chessboard_from_file(args.preprocess)
            logging.info(f"Prétraitement réussi. Affichage du résultat : {pprint.pformat(data)}")
        except ValueError as e:
            logging.info(f"Échec du prétraitement : {e}")
    elif args.fen is not None:
        try:
            data = model.predict_from_file(args.fen)
            logging.info(f"Prédiction réussie. Affichage du résultat : {data.board_fen()}")
        except ValueError as e:
            logging.info(f"Échec de la prédiction : {e}")
    elif args.bestmove is not None:
        move = best_move.get_best_move_from_fen(args.bestmove)
        if move is not None:
            logging.info(f"Stockfish a réussi. Affichage du résultat : {move.uci()}")
        else:
            logging.info("Stockfish a échoué.")
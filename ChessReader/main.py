#!/usr/bin/env python3

import argparse
import logging
logging.basicConfig(level = logging.INFO)
import pathlib
import src.board_reader.model as model
import src.server.server as server
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prépare l'image d'échiquier à être entraînée.")
    parser.add_argument("--input", type=pathlib.Path, help="Chemin vers l'image à être analysée.")
    parser.add_argument("--build", action="store_true", help="Entraîne le modèle avec les images de jeu dans 'neural-network-dataset'.")
    parser.add_argument("--train", type=int, help="Entraine le dataset le nombre de fois indiqué.")
    parser.add_argument("--server", type=int, help="Lance le serveur au port indiqué.")
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    if args.server is not None:
        server.start(args.server)
    elif args.input is not None:
        model.image(args.input)
    elif args.build:
        model.build_dataset_tree_structure()
    elif args.train is not None:
        model.train_model(args.train)
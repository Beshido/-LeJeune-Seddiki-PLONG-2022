#!/usr/bin/env python3

import argparse
import logging
logging.basicConfig(level = logging.INFO)
import pathlib
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prépare l'image d'échiquier à être entraînée.")
    parser.add_argument("--input", type=pathlib.Path, help="Chemin vers l'image à être analysée.")
    parser.add_argument("--train", action="store_true", help="Entraîne le modèle avec les images de jeu dans 'neural-network-dataset'.")
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    import board_reader.model
    if args.train:
        board_reader.model.build_and_train()
    
    elif args.input is not None:
        board_reader.model.image(args.input)
import src.board_reader.preprocess as preprocess
import chess.pgn
import cv2
import keras
from keras import layers
import logging
logging.basicConfig(level = logging.INFO)
import natsort
import numpy
import pathlib
import shutil
import tensorflow

CLASSES = [ "b", "empty", "k", "n", "p", "q", "r" ]

BASE_DIR = pathlib.Path("neural-network/")
DATASET_DIR = pathlib.Path(BASE_DIR, "dataset/")
OUTPUT_DIR = pathlib.Path(DATASET_DIR, "pieces/")
MODEL_LOCATION = pathlib.Path(BASE_DIR, "model/")

def build_dataset_tree_structure() -> None:
    """Coupe les images de toutes les parties d'échiquiers pour que seules les pièces soient visibles et soient dans le dossier approprié pour la construction de l'objet Dataset."""
    logging.info(f"Nettoyage du dossier {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    logging.info(f"Suppression du contenu du dossier {OUTPUT_DIR} réalisé avec succès.")
    
    unprocessable_images = []
    for game_dir in DATASET_DIR.iterdir():
        if not game_dir.is_dir():
            continue
        try:
            pgn_filename = next(game_dir.glob("*.pgn"))

        except StopIteration:
            continue

        logging.info(f"Lecture du fichier pgn suivant : {pgn_filename}")
        with pgn_filename.open() as pgn_file:
            game = chess.pgn.read_game(pgn_file)
        board = game.board()

        for move, chessboard_image in zip(game.mainline_moves(), natsort.natsorted(game_dir.glob("img/*.jpg"))):
            try:
                chessboard_image_prepreoccesed_data = preprocess.preprocess_chessboard(chessboard_image)
                chessboard_image_prepreoccesed_data.reverse()
                cropped_images = []
                tmp = []
                for index, item in enumerate(chessboard_image_prepreoccesed_data):
                    tmp.append(item[0])
                    if ((index + 1) % 8 == 0):
                        tmp.reverse()
                        cropped_images.extend(tmp)
                        tmp = []

                for i in range(64):
                    piece = board.piece_at(i)
                    corresponding_image = cropped_images[i]

                    save_location = pathlib.Path(OUTPUT_DIR, f"{piece.symbol().lower() if piece is not None else 'empty'}/{chessboard_image.stem}_{(i + 1):02d}{chessboard_image.suffix}")
                    save_location.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(save_location.resolve().as_posix(), corresponding_image)
                    chessboard_image.unlink()
                    print(
                        piece if piece is not None else ".", 
                        end="\n" if (i + 1) % 8 == 0 else " ",
                        flush=True
                    )

            except ValueError:
                logging.info(f"Échec du prétaitement de l'image suivante : {chessboard_image}")
                unprocessable_images.append(chessboard_image)
            
            board.push(move)

    unprocessable_images_log_message = "Les images suivantes n'ont pas pu être prétraitées :\n"
    for index, unprocessable_image in enumerate(unprocessable_images):
        unprocessable_images_log_message += f"\t{index}. {unprocessable_image}\n"
    logging.info(unprocessable_images_log_message)

def train_model(epochs: int = 10) -> None:
    train_datagen = tensorflow.keras.utils.image_dataset_from_directory(
        OUTPUT_DIR.resolve().as_posix(),
        class_names=CLASSES,
        image_size=(100, 100),
        validation_split= 0.2,
        subset="training",
        seed=123
    )
    validation_datagen = tensorflow.keras.utils.image_dataset_from_directory(
        OUTPUT_DIR.resolve().as_posix(),
        class_names=CLASSES,
        image_size=(100, 100),
        validation_split= 0.2,
        subset="validation",
        seed=123
    )

    try:
        model = keras.models.load_model(MODEL_LOCATION)

    except IOError:
        model = keras.models.Sequential([
            layers.Rescaling(1./255, input_shape=(100, 100, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(CLASSES))
        ])

    model.compile(optimizer="adam", loss=tensorflow.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.summary()
    history = model.fit(
        train_datagen,
        validation_data=validation_datagen,
        epochs=epochs
    )

    model.save(MODEL_LOCATION)

def image(image_file: pathlib.Path) -> list:
    preprocessed_data = preprocess.preprocess_chessboard(image_file)
    
    model = keras.models.load_model(MODEL_LOCATION)
    items = []
    for index, item in enumerate(preprocessed_data):
        image = item[0]
        image = cv2.resize(image, (100, 100))
        image_array = numpy.asarray(image)
        image_array = tensorflow.expand_dims(image_array, 0)

        predictions = model.predict(image_array)
        score = tensorflow.nn.softmax(predictions[0])

        logging.info(f"{index + 1}. Cette image appartient probablement à la classe {CLASSES[numpy.argmax(score)]} avec {100 * numpy.max(score):.2f}% de confiance.")
        items.append(CLASSES[numpy.argmax(score)])
        preprocess.show_image(image)

    logging.info(items)
    return items

def build_and_train():
    build_dataset_tree_structure()
    train_model()
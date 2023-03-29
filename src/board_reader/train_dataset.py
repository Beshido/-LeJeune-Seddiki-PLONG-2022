import chess.pgn, keras, pathlib

base_dir = pathlib.Path("neural-network/")

def label_images():
    dataset_dir = pathlib.Path(base_dir, "dataset/")
    for dir in dataset_dir.iterdir():
        if dir.is_dir():
            for pgn_filename in dir.glob("*.pgn"):
                with pgn_filename.open() as pgn_file:
                    game = chess.pgn.read_game(pgn_file)
                    board = game.board()

                    move = game.mainline_moves()
                    game.push(move)

                    for i in range(64):
                        piece = board.piece_at(i)
                        if piece is None:
                            pass
                        print(piece)


def train_dataset():
    train_dir = pathlib.Path(base_dir, "train/")
    validation_dir = pathlib.Path(base_dir, "validation/")
    test_dir = pathlib.Path(base_dir, "test/")

    train_datagen = keras.ImageDataGenerator(
        preprocess_input=keras.applications.inception_resnet_v2.preprocess_input,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True
    )
    
    test_datagen = keras.ImageDataGenerator(preprocess_input=keras.applications.inception_resnet_v2.preprocess_input)

if __name__ == "__main__":
    label_images()
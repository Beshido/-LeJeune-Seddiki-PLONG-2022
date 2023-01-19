import cv2, enum, logging

CHESSBOARD_SIZE = (7, 6)

class Color(enum.Enum):
    BLACK = 0
    WHITE = 1

# Setup logger
logging.basicConfig(level = logging.INFO)


def read_chessboard(image: cv2.Mat) -> list:
    for i in range(3, 9):
        for j in range(3, 9):
            ret, corners = cv2.findChessboardCorners(image, (i, j), None)
            if ret:
                logging.info(f"A chessboard was found with ({i}, {j}).")
                image = cv2.drawChessboardCorners(image, (i, j), corners, ret)

if __name__ == "__main__":

    logging.info("Launching script...")
    image = cv2.imread("img/chessboard-topview/image1.jpg")
    read_chessboard(image)
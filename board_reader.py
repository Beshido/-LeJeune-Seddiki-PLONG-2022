import cv2, enum, logging

CHESSBOARD_SIZE = (7, 7) # taille intérieure d'un échiquier
DELAY = 2000 # ms

logging.basicConfig(level = logging.INFO) # mise en place du logger

class Color(enum.Enum):
    BLACK = 0
    WHITE = 1

def show_image(image: cv2.Mat, delay: int = DELAY) -> None:
    if image is None: return

    cv2.imshow("show_image", image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def crop_to_square(image: cv2.Mat) -> cv2.Mat:
    height = image.shape[0]
    width = image.shape[1]

    size = width if height > width else height
    x1 = width // 2 - size // 2
    y1 = height // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    logging.info(f"Origin point is ({x1}, {y1}). Ending point is ({x2}, {y2}). The size of the square is {size}.")
    return image[y1:y2, x1:x2]


def read_chessboard(image: cv2.Mat) -> None:
    def find_contour(image: cv2.Mat) -> cv2.Mat:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

    def get_chessboard_corners_coordinates(image: cv2.Mat) -> list:
        ret, corners = cv2.findChessboardCornersSB(image, CHESSBOARD_SIZE, None)
        if ret:
            logging.info(f"A chessboard was found.")
        return corners

    image = crop_to_square(image)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = get_chessboard_corners_coordinates(image_bw)

    for i in range(len(corners) - 8):
        x1 = int(corners[i][0][0])
        y1 = int(corners[i][0][1])
        x2 = int(corners[i + 8][0][0])
        y2 = int(corners[i + 8][0][1])

        if x1 > x2 or y1 > y2: continue

        logging.info(f"{i}. Origin point is ({x1}, {y1}). Ending point is ({x2}, {y2}). Image size is {image.shape[1]}x{image.shape[0]}.")
        show_image(image[y1:y2, x1:x2])

if __name__ == "__main__":
    logging.info("Launching script...")
    image = cv2.imread("img/chessboard-topview/image1.jpg")
    read_chessboard(image)
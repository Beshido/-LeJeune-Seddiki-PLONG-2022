import cv2, enum, logging

CHESSBOARD_SIZE = (7, 7) # taille intérieure d'un échiquier
DELAY = 2000 # ms

logging.basicConfig(level = logging.INFO) # mise en place du logger

class Color(enum.Enum):
    BLACK = 0
    WHITE = 1

def show_image(image: cv2.Mat):
    cv2.imshow("test", image)
    cv2.waitKey(DELAY)
    cv2.destroyAllWindows()

def read_chessboard(image: cv2.Mat) -> list:
    def findContour(image: cv2.Mat) -> cv2.Mat:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

    def findChessoard(image: cv2.Mat):
        for i in range(3, 9):
            for j in range(3, 9):
                ret, corners = cv2.findChessboardCornersSB(image, (i, j), None)
                if ret:
                    logging.info(f"A chessboard was found with ({i}, {j}).")
                    image = cv2.drawChessboardCorners(image, (i, j), corners, ret)
                    cv2.imshow("read_chessboard", image)
                    cv2.waitKey(DELAY)
        cv2.destroyAllWindows()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(findContour(image))

if __name__ == "__main__":
    logging.info("Launching script...")
    image = cv2.imread("img/chessboard-topview/image1.jpg")
    read_chessboard(image)
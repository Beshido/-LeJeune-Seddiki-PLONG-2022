import cv2, enum, logging, pprint

CHESSBOARD_SIZE = (7, 7) # taille intérieure d'un échiquier
DELAY = 2000 # ms

logging.basicConfig(level = logging.INFO) # mise en place du logger

class Color(enum.Enum):
    BLACK = 0
    WHITE = 1

class Coordinates:
    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self) -> str:
        return f"{self.x1}, {self.y1} ; {self.x2}, {self.y2}"

class PiecesType(enum.Enum):
    PAWN = 8
    ROOK = 2
    BISHOP = 2
    KNIGHT = 2
    QUEEN = 1
    KING = 1

class Piece:
    def __init__(self, image: cv2.Mat) -> None:
        self.image = image

class Board:
    def __init__(self) -> None:
        pass

def show_image(image: cv2.Mat, delay: int = DELAY) -> None:
    if image is None: return

    cv2.imshow("show_image", image)
    cv2.waitKey(delay)

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


def read_chessboard(image: cv2.Mat) -> list[list[Coordinates]]:
    image = crop_to_square(image)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE, None)
    if not ret:
        return []
    
    logging.info(f"A chessboard was found.")
    assert(len(corners) == 49)

    coordinates = [[]]
    for i in range(len(corners) - 8):
        
        x1 = int(corners[i][0][0])
        y1 = int(corners[i][0][1])
        x2 = int(corners[i + 8][0][0])
        y2 = int(corners[i + 8][0][1])

        if x1 > x2 or y1 > y2: 
            coordinates.append([])
            continue

        logging.info(f"{i}. Origin point is ({x1}, {y1}). Ending point is ({x2}, {y2}). Image size is {image.shape[1]}x{image.shape[0]}.")
        coordinates[-1].append(Coordinates(x1, y1, x2, y2))
    
    # coordonnées extérieures
    for item in coordinates:
        point_first = item[0]
        point_second = item[1]
        point_before_last = item[-2]
        point_last = item[-1]

        x1_offset_first = point_second.x1 - point_first.x1
        y1_offset_first = point_second.y1 - point_first.y1
        x2_offset_first = point_second.x2 - point_first.x2
        y2_offset_first = point_second.y2 - point_first.y2
        point0 = Coordinates(point_first.x1 - x1_offset_first, point_first.y1 - y1_offset_first, point_first.x2 - x2_offset_first, point_first.y2 - y2_offset_first)
        
        x1_offset_last = point_last.x1 - point_before_last.x1
        y1_offset_last = point_last.y1 - point_before_last.y1
        x2_offset_last = point_last.x2 - point_before_last.x2
        y2_offset_last = point_last.y2 - point_before_last.y2
        point8 = Coordinates(point_last.x1 + x1_offset_last, point_last.y1 + y1_offset_last, point_last.x2 + x2_offset_last, point_last.y2 + y2_offset_last)

        item.insert(0, point0)
        item.append(point8)
    coordinates.insert(0, [])
    coordinates.append([])
    for i in range(len(coordinates)):
        point_first = coordinates[1][i]
        point_second = coordinates[2][i]
        point_before_last = coordinates[-3][i]
        point_last = coordinates[-2][i]

        x1_offset_first = point_second.x1 - point_first.x1
        y1_offset_first = point_second.y1 - point_first.y1
        x2_offset_first = point_second.x2 - point_first.x2
        y2_offset_first = point_second.y2 - point_first.y2
        point0 = Coordinates(point_first.x1 - x1_offset_first, point_first.y1 - y1_offset_first, point_first.x2 - x2_offset_first, point_first.y2 - y2_offset_first)

        x1_offset_last = point_last.x1 - point_before_last.x1
        y1_offset_last = point_last.y1 - point_before_last.y1
        x2_offset_last = point_last.x2 - point_before_last.x2
        y2_offset_last = point_last.y2 - point_before_last.y2
        point8 = Coordinates(point_last.x1 + x1_offset_last, point_last.y1 + y1_offset_last, point_last.x2 + x2_offset_last, point_last.y2 + y2_offset_last)

        coordinates[0].append(point0)
        coordinates[-1].append(point8)

    return coordinates

def get_pieces_location(image: cv2.Mat) -> list:
    
    return []
 
if __name__ == "__main__":
    logging.info("Launching script...")
    image = cv2.imread("img/chessboard-topview/image3.webp")
    pprint.pprint(read_chessboard(image))

    cv2.destroyAllWindows()
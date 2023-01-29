import cv2, enum, logging, pprint

CHESSBOARD_SIZE = (7, 7) # taille intérieure d'un échiquier
DELAY = 2000 # ms

logging.basicConfig(level = logging.INFO) # mise en place du logger

class Color(enum.Enum):
    BLACK = 0
    WHITE = 1

class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = int(x)
        self.y = int(y)

    def __add__(self, point):
        return Point(self.x + point.x, self.y + point.y)
    
    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

class Coordinates:
    def __init__(self, upperleft: Point, upperright: Point, lowerleft: Point, lowerright: Point) -> None:
        self.upperleft = upperleft
        self.upperright = upperright
        self.lowerleft = lowerleft
        self.lowerright = lowerright

    def __add__(self, coordinate):
        return Coordinates(self.upperleft + coordinate.upperleft, self.upperright + coordinate.upperright, self.lowerleft + coordinate.lowerleft, self.lowerright + coordinate.lowerright)

    def __sub__(self, coordinate):
        return Coordinates(self.upperleft - coordinate.upperleft, self.upperright - coordinate.upperright, self.lowerleft - coordinate.lowerleft, self.lowerright - coordinate.lowerright)


    def __repr__(self) -> str:
        return f"{self.upperleft}, {self.upperright} ; {self.lowerleft}, {self.lowerright}"
    
    def get_image(self, image: cv2.Mat) -> cv2.Mat:
        return image[self.upperleft.y:self.lowerright.y, self.upperleft.x:self.lowerright.y]

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


def read_chessboard(image: cv2.Mat) -> list: # list[list[Coordinates]]
    image = crop_to_square(image)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE, None)
    if not ret:
        return []
    
    logging.info(f"A chessboard was found.")
    assert(len(corners) == 49)

    coordinates = [[]]
    for i in range(len(corners) - 8):
        upperleft = Point(corners[i][0][0], corners[i][0][1])
        upperright = Point(corners[i + 1][0][0], corners[i + 1][0][1])
        lowerleft = Point(corners[i + 7][0][0], corners[i + 7][0][1])
        lowerright = Point(corners[i + 8][0][0], corners[i + 8][0][1])

        if upperleft.x > lowerright.x or upperleft.y > lowerright.y: 
            coordinates.append([])
            continue

        logging.info(f"{i}. Origin point is {upperleft}. Ending point is {lowerright}). Image size is {image.shape[1]}x{image.shape[0]}.")
        coordinates[-1].append(Coordinates(upperleft, upperright, lowerleft, lowerright))
    
    # coordonnées extérieures TODO convertir en vrai coordinates
    for item in coordinates:
        point_first = item[0]
        point_second = item[1]
        point_before_last = item[-2]
        point_last = item[-1]

        point0 = point_first - (point_second - point_first)
        point8 = point_last + (point_last - point_before_last)

        item.insert(0, point0)
        item.append(point8)
    coordinates.insert(0, [])
    coordinates.append([])
    for i in range(len(coordinates)):
        point_first = coordinates[1][i]
        point_second = coordinates[2][i]
        point_before_last = coordinates[-3][i]
        point_last = coordinates[-2][i]

        point0 = point_first - (point_second - point_first)
        point8 = point_last + (point_last - point_before_last)

        coordinates[0].append(point0)
        coordinates[-1].append(point8)

    for row in coordinates:
        for coordinate in row:
            show_image(coordinate.get_image(image))
    return coordinates

def get_pieces_location(image: cv2.Mat) -> list:
    
    return []
 
if __name__ == "__main__":
    logging.info("Launching script...")
    image = cv2.imread("img/chessboard-topview/image3.webp")
    pprint.pprint(read_chessboard(image))

    cv2.destroyAllWindows()
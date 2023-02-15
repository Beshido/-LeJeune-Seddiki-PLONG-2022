import cv2, enum, logging, matplotlib, pprint, numpy
# from matplotlib import pyplot
# matplotlib.use('TkAgg')
# logging.basicConfig(level = logging.INFO) # mise en place du logger

CHESSBOARD_SIZE = (7, 7) # taille intérieure d'un échiquier
HIST_THRESHOLD = 3
CROP_VALUE = 10
MIN_PEAKS = 5 # nombre minimum de peak dans l'histogramme pour que la case soit considérée comme remplie

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

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

class Coordinates:
    def __init__(self, upperleft: Point, upperright: Point, lowerleft: Point, lowerright: Point) -> None:
        self.upperleft = upperleft
        self.upperright = upperright
        self.lowerleft = lowerleft
        self.lowerright = lowerright
    
    def minX(self) -> int:
        return min(self.upperleft.x, self.upperright.x, self.lowerleft.x, self.lowerright.x)

    def minX1(self) -> int:
        return max(self.upperleft.x, self.lowerleft.x) + CROP_VALUE
    
    def minX2(self) -> int:
        return min(self.upperright.x, self.lowerright.x) - CROP_VALUE

    def minY1(self) -> int:
        return max(self.upperleft.y, self.upperright.y) + CROP_VALUE

    def minY2(self) -> int:
        return min(self.lowerleft.y, self.lowerright.y) - CROP_VALUE

    def maxX(self) -> int:
        return max(self.upperleft.x, self.upperright.x, self.lowerleft.x, self.lowerright.x)

    def minY(self) -> int:
        return min(self.upperleft.y, self.upperright.y, self.lowerleft.y, self.lowerright.y)
    
    def maxY(self) -> int:
        return max(self.upperleft.y, self.upperright.y, self.lowerleft.y, self.lowerright.y)

    def get_image(self, image: cv2.Mat) -> cv2.Mat:
        return image[self.minY1():self.minY2(), self.minX1():self.minX2()]

    def __add__(self, coordinate):
        return Coordinates(self.upperleft + coordinate.upperleft, self.upperright + coordinate.upperright, self.lowerleft + coordinate.lowerleft, self.lowerright + coordinate.lowerright)

    def __sub__(self, coordinate):
        return Coordinates(self.upperleft - coordinate.upperleft, self.upperright - coordinate.upperright, self.lowerleft - coordinate.lowerleft, self.lowerright - coordinate.lowerright)

    def __str__(self) -> str:
        return f"[ {self.upperleft}, {self.upperright}\n  {self.lowerleft}, {self.lowerright} ]"

class PiecesType(enum.Enum):
    PAWN = 6
    ROOK = 5
    BISHOP = 4
    KNIGHT = 3
    QUEEN = 2
    KING = 1
    UNKNOWN = 0

    def __str__(self) -> str:
        return self.name

class Piece:
    def __init__(self, image: cv2.Mat, type: PiecesType, filled: bool) -> None:
        self.image = image
        self.type = type
        self.filled = filled

    def __str__(self) -> str:
        return str(self.type)
    
    def __repr__(self) -> str:
        return str(self)

class Board:
    def __init__(self) -> None:
        pass

def show_image(image: cv2.Mat) -> None:
    if image is None: return

    cv2.imshow("show_image", image)
    cv2.waitKey(10000)

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

def get_number_peaks(hist: cv2.Mat) -> int:
    n = 0
    for i in range(1, len(hist) - 1):
        value = hist[i][0] - HIST_THRESHOLD
        if value > hist[i - 1][0] and value > hist[i + 1][0]:
            n += 1
    return n

def get_cases_coordinates(image_path: str) -> list:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"The given image path does not exist: '{image_path}'")

    image = crop_to_square(image)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(image_bw)
    ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE, None)
    if not ret:
        raise ValueError("OpenCV2 was unable to find (7, 7) chessboard-like patterns.")
    
    logging.info(f"A chessboard was found.")
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
    
    return coordinates

def get_cases_coordinates_harris(image_path: str) -> list:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = numpy.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    threshold = 0.7
    img[dst > threshold * dst.max()] = [0, 0, 255]
    num_corners = numpy.sum(dst > threshold * dst.max())
    show_image(img)

def get_cases_coordinates_neural(image: cv2.Mat) -> list:
    pass

def check_cases_content(image_path: str, coordinates: list) -> list:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"The given image path does not exist: '{image_path}'")

    image = crop_to_square(image)
    contrast = 0.5
    brightness = 10
    blur_factor = (5, 5)
    modified_image = cv2.blur(cv2.convertScaleAbs(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) , alpha = contrast, beta = brightness), blur_factor)
    board = []
    for row in coordinates:
        board.append([])
        for coordinate in row:
            img = coordinate.get_image(modified_image)

            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            peak = get_number_peaks(hist)
            if peak < MIN_PEAKS:
                board[-1].append(Piece(coordinate.get_image(image), PiecesType.UNKNOWN, False))
            else:
                board[-1].append(Piece(coordinate.get_image(image), PiecesType.UNKNOWN, True))
                                
    return board

def chessboard_piece() -> list:
    pass

if __name__ == "__main__":
    logging.info("Launching script...")
    image = "img/chessboard-topview/image4.jpg"
    get_cases_coordinates_harris(image)

    cv2.destroyAllWindows()
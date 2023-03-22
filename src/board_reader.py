import cv2, enum, logging, matplotlib, pprint, numpy, sys

CHESSBOARD_SIZE = (7, 7) # taille intérieure d'un échiquier
HIST_THRESHOLD = 3
CROP_VALUE = 10
MIN_PEAKS = 10 # nombre minimum de peak dans l'histogramme pour que la case soit considérée comme remplie

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
    
    def __repr__(self) -> str:
        return str(self)
    
    def __iter__(self):
        return iter([self.x, self.y])

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
    
    def __repr__(self) -> str:
        return str(self)

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
        return str(self.type) if self.filled else "EMPTY"
    
    def __repr__(self) -> str:
        return str(self)

def get_number_peaks(hist: cv2.Mat) -> int:
    n = 0
    for i in range(1, len(hist) - 1):
        value = hist[i][0] - HIST_THRESHOLD
        if value > hist[i - 1][0] and value > hist[i + 1][0]:
            n += 1
    return n

def get_cases_coordinates(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de coordonnées de cases d'échiquier via la méthode de findChessboardCornersSB de OpenCV. Efficace pour les images avec une vue de haut."""
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE, None)
    if not ret:
        raise ValueError(f"OpenCV2 was unable to find {CHESSBOARD_SIZE} chessboard-like patterns.")
    
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

        logging.debug(f"{i}. Origin point is {upperleft}. Ending point is {lowerright}). Image size is {image.shape[1]}x{image.shape[0]}.")
        coordinates[-1].append(Coordinates(upperleft, upperright, lowerleft, lowerright))
    
    # coordonnées extérieures
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

def get_cases_coordinates_harris(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de coordonnées de cases d'échiquier via la méthode de Harris de OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = numpy.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    threshold = 0.7
    image[dst > threshold * dst.max()] = [0, 0, 255]
    num_corners = numpy.sum(dst > threshold * dst.max())

def wrap_image(image: cv2.Mat, coordinates: list) -> cv2.Mat:
    """Distords l'image pour que celle-ci ne contienne uniquement les cases du plateau de jeu."""
    height = image.shape[0]
    width = image.shape[1]
    length = int(height) if height > width else int(width)
    
    topleft = list(coordinates[0][0].upperleft)
    topright = list(coordinates[0][-1].upperright)
    bottomleft = list(coordinates[-1][0].lowerleft)
    bottomright = list(coordinates[-1][-1].lowerright)

    pts1 = numpy.float32([topleft, topright, bottomleft, bottomright])
    pts2 = numpy.float32([(0, 0), (length, 0), (0, length), (length, length)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (length, length))

    h = image.shape[0] / 8
    w = image.shape[1] / 8
    for i in range(8):
        for j in range(8):
            c = Coordinates(Point(i * w, j * h), Point((i + 1) * w, j * h), Point(i * w, (j + 1) * h), Point((i + 1) * w, (j + 1) * h))

    return dst

def check_cases_content(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de pièces d'échiquier, précisant si oui ou non chaque case est vide ou non."""
    contrast = 0.5
    brightness = 10
    blur_factor = (5, 5)
    modified_image = cv2.blur(cv2.convertScaleAbs(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) , alpha = contrast, beta = brightness), blur_factor)
    board = []
    peaks = []
    h = image.shape[0] / 8
    w = image.shape[1] / 8
    for i in range(8):
        board.append([])
        peaks.append([])
        for j in range(8):
            c = Coordinates(Point(i * w, j * h), Point((i + 1) * w, j * h), Point(i * w, (j + 1) * h), Point((i + 1) * w, (j + 1) * h))
            img = c.get_image(modified_image)

            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            peak = get_number_peaks(hist)
            empty = peak < MIN_PEAKS

            board[-1].append(not empty)
            peaks[-1].append(peak)
            
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                cv2.imshow(f"{i}.{j}: {peak}", img)
                cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    if logging.getLogger().isEnabledFor(logging.INFO):
        pprint.pprint(board)
        pprint.pprint(peaks)
        

    return board

def image_to_chessboard(image_path: str) -> list:
    """Méthode maîtresse qui convertit une image en échiquier digital."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"The given image path does not exist: '{image_path}'")

    coordinates = get_cases_coordinates(image)
    warped_image = wrap_image(image, coordinates)
    pieces = check_cases_content(warped_image)
    return pieces

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    logging.info("Launching script...")

    if (len(sys.argv) > 1):
        image_path = sys.argv[1]
    else:
        logging.info("Missing input image argument.")
        exit(0)

    image_to_chessboard(image_path)
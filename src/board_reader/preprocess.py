import cv2
import logging
logging.basicConfig(level = logging.INFO)
import pathlib
import numpy

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

    def minX1(self) -> int:
        return max(self.upperleft.x, self.lowerleft.x) + CROP_VALUE
    
    def minX2(self) -> int:
        return min(self.upperright.x, self.lowerright.x) - CROP_VALUE

    def minY1(self) -> int:
        return max(self.upperleft.y, self.upperright.y) + CROP_VALUE

    def minY2(self) -> int:
        return min(self.lowerleft.y, self.lowerright.y) - CROP_VALUE

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

def show_image(image: cv2.Mat) -> None:
    if image is None: return

    cv2.imshow("show_image", image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        logging.info("On s'arrête là.")
        exit(0)

def crop_to_square(image: cv2.Mat) -> cv2.Mat:
    """Transforme une image en la rognant sur le centre, afin de supprimer en partie le background de l'image."""
    height = image.shape[0]
    width = image.shape[1]

    size = width if height > width else height
    x1 = width // 2 - size // 2
    y1 = height // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    return image[y1:y2, x1:x2]

def get_cases_coordinates(image: cv2.Mat) -> list:
    """Renvoie une liste de coordonnées de cases d'échiquier via la méthode de findChessboardCornersSB de OpenCV. Efficace pour les images avec une vue de haut."""
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(image_bw, CHESSBOARD_SIZE)
    if not ret:
        ret, corners = cv2.findChessboardCorners(image_bw, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if not ret:
            ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE)
            if not ret:
                ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_EXHAUSTIVE)
                if not ret:
                    raise ValueError(f"OpenCV2 was unable to find {CHESSBOARD_SIZE} chessboard-like patterns.")

    coordinates = [[]]
    for i in range(len(corners) - 8):
        upperleft = Point(corners[i][0][0], corners[i][0][1])
        upperright = Point(corners[i + 1][0][0], corners[i + 1][0][1])
        lowerleft = Point(corners[i + 7][0][0], corners[i + 7][0][1])
        lowerright = Point(corners[i + 8][0][0], corners[i + 8][0][1])

        if upperleft.x > lowerright.x or upperleft.y > lowerright.y: 
            coordinates.append([])
            continue

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

    return sum(coordinates, []) # aplatit la liste

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
    return num_corners

def get_chessboard_outline(image: cv2.Mat) -> list:
    """Renvoie une liste contenant les coordonnées des contours de l'image d'un échiquier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = 255 - morph
    cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cntr_img = numpy.zeros_like(morph)
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 200: 
            cv2.drawContours(cntr_img, [c], 0, 255, 1)

    # get all non-zero points
    points = numpy.column_stack(numpy.where(cntr_img.transpose() > 0))
    hull = cv2.convexHull(points)

    result = image.copy()
    cv2.polylines(result, [hull], True, (0,0,255), 2)

def wrap_image(image: cv2.Mat, coordinates: list) -> cv2.Mat:
    """Distords l'image pour que celle-ci ne contienne uniquement les cases du plateau de jeu."""
    height = image.shape[0]
    width = image.shape[1]
    length = int(height) if height > width else int(width)

    topleft = list(coordinates[0].upperleft)
    topright = list(coordinates[7].upperright)
    bottomleft = list(coordinates[56].lowerleft)
    bottomright = list(coordinates[63].lowerright)

    pts1 = numpy.float32([topleft, topright, bottomleft, bottomright])
    pts2 = numpy.float32([(0, 0), (length, 0), (0, length), (length, length)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (width, height))

    h = height / 8
    w = width / 8
    for i in range(8):
        for j in range(8):
            c = Coordinates(Point(i * w, j * h), Point((i + 1) * w, j * h), Point(i * w, (j + 1) * h), Point((i + 1) * w, (j + 1) * h))

    return dst

def get_cases_color(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de couleurs de cases d'échiquier. True pour blanc, False pour noir."""
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = bw_image.shape[0] // 8, bw_image.shape[1] // 8
    blur = cv2.GaussianBlur(bw_image, (5, 5), 0)
    _, image_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x, y = 0, 0
    lowest_color_coordinates = (None, None)
    lowest_color_content = None
    for x_index in range(8):
        for y_index in range(8):
            coordinate = Coordinates(
                Point(x, y), 
                Point(x + width, y), 
                Point(x, y + height), 
                Point(x + width, y + height)
            )
            average_color = numpy.average(coordinate.get_image(image_binary))
            if lowest_color_content is None or average_color < lowest_color_content:
                lowest_color_content = average_color
                color = average_color > 255 // 2
                lowest_color_coordinates = (x_index, y_index)

            y += height
        x += width
        y = 0

    if lowest_color_content == 255:
        raise ValueError("Couldn't get the chessboard cases color.")
            
    colors = []
    current_color = color if ((lowest_color_coordinates[0] % 2 == 0 and lowest_color_coordinates[1] == 0) or (lowest_color_coordinates[0] % 2 != 0 and lowest_color_coordinates[1] % 2 != 0)) else not color
    for _ in range(8):
        for _ in range(8):
            colors.append(current_color)
            current_color = not current_color
        current_color = not current_color
    return colors

def check_cases_content(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de booléen représentant si une case est vide ou pas."""

    def get_number_peaks(hist: cv2.Mat) -> int:
        n = 0
        for i in range(1, len(hist) - 1):
            value = hist[i][0] - HIST_THRESHOLD
            if value > hist[i - 1][0] and value > hist[i + 1][0]:
                n += 1
        return n

    image = crop_to_square(image)
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

        cv2.destroyAllWindows()

    return board

def preprocess_chessboard(image_path: pathlib.Path) -> list:
    """Méthode maîtresse qui convertit une image en échiquier digital. Renvoie une exception ValueError si l'image n'est pas valide ou si OpenCV échoue la reconnaissance de l'échiquier."""
    logging.info(f"Prétraitement de l'image suivante en cours : {image_path}")
    image = cv2.imread(image_path.resolve().as_posix())
    if image is None:
        raise ValueError(f"Le chemin vers l'image suivante n'exsite pas : '{image_path}'")

    image = crop_to_square(image)
    coordinates = get_cases_coordinates(image)
    warped_image = wrap_image(image, coordinates)
    colors = get_cases_color(warped_image)
    # pieces = check_cases_content(warped_image)
    output = []
    for coordinate, case_color in zip((coordinates), colors):
        output.append((coordinate.get_image(image), case_color))

    assert len(output) == 64
    logging.info(f"Succès du prétraitement de l'image suivante : {image_path}")
    return output
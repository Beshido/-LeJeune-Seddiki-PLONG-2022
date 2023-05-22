import cv2
import logging
logging.basicConfig(level = logging.INFO)
import pathlib
import numpy

CHESSBOARD_SIZE = (7, 7)
"""Taille intérieure d'un échiquier."""

HIST_THRESHOLD = 3
"""Seuil de l'histogramme pour déterminer si une case est vide ou non."""

CROP_VALUE = 10
"""Valeur de rognage de l'image."""

MIN_PEAKS = 10
"""Nombre minimum de peak dans l'histogramme pour que la case soit considérée comme contenant une case."""

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
        return max(self.upperleft.x, self.lowerleft.x)
    
    def minX2(self) -> int:
        return min(self.upperright.x, self.lowerright.x)

    def minY1(self) -> int:
        return max(self.upperleft.y, self.upperright.y)

    def minY2(self) -> int:
        return min(self.lowerleft.y, self.lowerright.y)

    def maxX1(self) -> int:
        return min(self.upperleft.x, self.lowerleft.x)
    
    def maxX2(self) -> int:
        return max(self.upperright.x, self.lowerright.x)

    def maxY1(self) -> int:
        return min(self.upperleft.y, self.upperright.y)

    def maxY2(self) -> int:
        return max(self.lowerleft.y, self.lowerright.y)

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

def _show_image(image: cv2.Mat) -> None:
    """Affiche une image dans une fenêtre. Appuyer sur une touche pour continuer le programme. Appuyer sur 'q' pour terminer le programme."""

    if image is None: return

    cv2.imshow("show_image", image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        logging.info("On s'arrête là.")
        exit(0)

def _crop_to_square(image: cv2.Mat) -> cv2.Mat:
    """Transforme une image en la rognant sur le centre, afin de supprimer en partie le background de l'image."""

    height = image.shape[0]
    width = image.shape[1]

    size = width if height > width else height
    x1 = width // 2 - size // 2
    y1 = height // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    return image[y1:y2, x1:x2]

def _get_cases_coordinates(image: cv2.Mat) -> list:
    """Renvoie une liste de coordonnées de cases d'échiquier via la méthode de findChessboardCornersSB de OpenCV. Efficace pour les images avec une vue de haut."""

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(image_bw, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER)
    if not ret:
        logging.info("Echec de la reconnaissance de l'échiquier avec la méthode findChessboardCornersSB. Tentative avec la méthode findChessboardCorners...")
        ret, corners = cv2.findChessboardCorners(image_bw, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            raise ValueError(f"OpenCV2 n'a pas réussi à trouver un pattern d'échiquier de {CHESSBOARD_SIZE}.")

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

def _get_cases_coordinates_harris(image: cv2.Mat) -> list:
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

def _get_chessboard_outline(image: cv2.Mat) -> list:
    """Renvoie une liste contenant les coordonnées des contours de l'image d'un échiquier."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

def _wrap_image(image: cv2.Mat, coordinates: list) -> cv2.Mat:
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

def _get_cases_color(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de couleurs de cases d'échiquier. True pour blanc, False pour noir."""

    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = bw_image.shape[0] // 8, bw_image.shape[1] // 8
    blur = cv2.GaussianBlur(bw_image, (5, 5), 0)
    _, image_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowest_color_coordinates = (None, None)
    lowest_color_content = None
    for x in range(8):
        for y in range(8):
            coordinate = Coordinates(
                Point(x * width, y * height), 
                Point(x * width + width, y * height), 
                Point(x * width, y * height + height), 
                Point(x * width + width, y * height + height)
            )
            average_color = numpy.average(coordinate.get_image(image_binary))
            if lowest_color_content is None or average_color < lowest_color_content:
                lowest_color_content = average_color
                color = average_color > 255 // 2
                lowest_color_coordinates = (x, y)

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

def _check_cases_content(image: cv2.Mat) -> list:
    """Renvoie une liste de liste de booléen représentant si une case est vide ou pas."""

    def get_number_peaks(hist: cv2.Mat) -> int:
        """Renvoie le nombre de peaks dans l'histogramme."""
    
        n = 0
        for i in range(1, len(hist) - 1):
            value = hist[i][0] - HIST_THRESHOLD
            if value > hist[i - 1][0] and value > hist[i + 1][0]:
                n += 1
        return n

    image = _crop_to_square(image)
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

def _get_pieces_color(image: cv2.Mat) -> list:
    """Renvoie une liste de booléen renvoyant True si la couleur de la pièce est l'opposé de la couleur de la case."""

    height, width = image.shape[0] // 8, image.shape[1] // 8
    blur = cv2.GaussianBlur(image, (5,5), 0)
    _, img_binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    img_binary_inverted = cv2.bitwise_not(img_binary)

    morph_kernel = numpy.ones((15, 15), numpy.uint8)
    output = cv2.morphologyEx(img_binary_inverted, cv2.MORPH_CLOSE, morph_kernel)
    pieces_color = []
    for x in range(8):
        for y in range(8):
            coordinate = Coordinates(
                Point(x * width, y * height), 
                Point(x * width + width, y * height), 
                Point(x * width, y * height + height), 
                Point(x * width + width, y * height + height)
            )
            average_color = numpy.average(coordinate.get_image(output))
            b = average_color >= 255 - CROP_VALUE or average_color <= 0 + CROP_VALUE
            pieces_color.append(b)
    return pieces_color

def _preprocess_chessboard(image: cv2.Mat, rotation_factor: int = 0) -> list:
    """Méthode maîtresse qui convertit une image en échiquier digital. Renvoie une exception ValueError si l'image n'est pas valide ou si OpenCV échoue la reconnaissance de l'échiquier. Format de retour : [ (image, case_color, piece_color), ... ]"""

    for _ in range(rotation_factor):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_factor > 0:
        logging.info(f"Rotation de l'image de {rotation_factor * 90} degrés réalisée avec succès.")
    image = cv2.flip(image, 1)
    coordinates = _get_cases_coordinates(image)
    logging.info(f"Récupération des coordonnées des cases de l'échiquier réalisée avec succès.")
    warped_image = _wrap_image(image, coordinates)
    logging.info(f"Distorsion de l'image réalisée avec succès.")
    colors = _get_cases_color(warped_image)
    logging.info(f"Récupération des couleurs des cases de l'échiquier réalisée avec succès.")
    pieces_color = _get_pieces_color(warped_image)
    logging.info(f"Récupération des couleurs des pièces de l'échiquier réalisée avec succès.")
    # pieces = check_cases_content(warped_image)
    output = []
    for coordinate, case_color, piece_color in zip((coordinates), colors, pieces_color):
        output.append((coordinate.get_image(image), case_color, piece_color))
        # _show_image(coordinate.get_image(image))

    assert len(output) == 64
    logging.info(f"Succès du prétraitement de l'image.")
    return output

def preprocess_chessboard_from_file(image_path: pathlib.Path, rotation_factor: int = 0) -> list:
    """Méthode maîtresse qui convertit un fichier en échiquier digital. Renvoie une exception ValueError si l'image n'est pas valide ou si OpenCV échoue la reconnaissance de l'échiquier."""

    logging.info(f"Prétraitement de l'image suivante en cours : {image_path}")
    image = cv2.imread(image_path.resolve().as_posix())
    if image is None:
        raise ValueError(f"Le chemin vers l'image suivante n'exsite pas : '{image_path}'")

    return _preprocess_chessboard(image, rotation_factor)

def preprocess_chessboard_from_memory(image: bytes, rotation_factor: int = 0) -> list:
    """Méthode maîtresse qui convertit une image en mémoire en échiquier digital. Renvoie une exception ValueError si l'image n'est pas valide ou si OpenCV échoue la reconnaissance de l'échiquier."""

    image = numpy.frombuffer(image, numpy.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Le buffer de l'image est illégal.")
    logging.info(f"L'image en mémoire a été chargée avec succès.")
    
    return _preprocess_chessboard(image, rotation_factor)

def preprocess_with_coordinates(image_path: pathlib.Path, coordinates: list, rotation_factor: int = 0) -> list:
    """Méthode maîtresse qui convertit une image en échiquier digital. Renvoie une exception ValueError si l'image n'est pas valide ou si OpenCV échoue la reconnaissance de l'échiquier. Format de retour : [ (image, case_color, piece_color), ... ]"""

    logging.info(f"Prétraitement de l'image suivante en cours : {image_path}")
    image = cv2.imread(image_path.resolve().as_posix())
    if image is None:
        raise ValueError(f"Le chemin vers l'image suivante n'exsite pas : '{image_path}'")
    for _ in range(rotation_factor):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    warped_image = _wrap_image(image, coordinates)
    logging.info(f"Distorsion de l'image réalisée avec succès.")
    colors = _get_cases_color(warped_image)
    logging.info(f"Récupération des couleurs des cases de l'échiquier réalisée avec succès.")
    pieces_color = _get_pieces_color(warped_image)
    logging.info(f"Récupération des couleurs des pièces de l'échiquier réalisée avec succès.")
    # pieces = check_cases_content(warped_image)
    output = []
    logging.info(image)
    for coordinate, case_color, piece_color in zip((coordinates), colors, pieces_color):
        output.append((coordinate.get_image(image), case_color, piece_color))
        # _show_image(coordinate.get_image(image))

    assert len(output) == 64
    logging.info(f"Succès du prétraitement de l'image.")
    return output
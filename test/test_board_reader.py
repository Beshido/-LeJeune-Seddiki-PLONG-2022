import cv2, unittest
from src.board_reader.preprocess import get_cases_coordinates, image_to_chessboard

def checkboard_test(chessboard: list, paths: list) -> bool:
    
    return True

class TestBoardReader(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def test_checkboard_recogniton(self):
        paths = [
            "img/chessboard-topview/image1.jpg",
            # "img/chessboard-topview/image2.png",
            "img/chessboard-topview/image3.webp",
            "img/chessboard-topview/image4.jpg",
            "img/chessboard-topview/image5.jpg",
            # "img/chessboard-topview/image6.jpg",
            # "img/chessboard-topview/image7.jpg"
        ]
        for path in paths:
            try:
                get_cases_coordinates(cv2.imread(path))
            except ValueError:
                self.fail(f"Chessboard recognition failed for '{path}'")

    def test_image_horizontal_init(self):
        initial_chessboard = [
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
            [ True, True, False, False, False, False, True, True ],
        ]
        paths = [
            "img/chessboard-topview/image1.jpg", # validé
            # "img/chessboard-topview/image2.png",
            "img/chessboard-topview/image3.webp", # validé
            # "img/chessboard-topview/image4.jpg"
        ]
        for path in paths:
            board = image_to_chessboard(path)
            self.assertEqual(initial_chessboard, board, f"Chessboard piece placement is different than expected for '{path}'")
        
    def test_image_vertical_init(self):
        initial_chessboard = [
            [ True, True, True, True, True, True, True, True ],
            [ True, True, True, True, True, True, True, True ],
            [ False, False, False, False, False, False, False, False ],
            [ False, False, False, False, False, False, False, False ],
            [ False, False, False, False, False, False, False, False ],
            [ False, False, False, False, False, False, False, False ],
            [ True, True, True, True, True, True, True, True ],
            [ True, True, True, True, True, True, True, True ],
        ]
        paths = [
            "img/chessboard-topview/image5.jpg", # validé
            # "img/chessboard-topview/image6.jpg",
            # "img/chessboard-topview/image7.jpg"
        ]
        for path in paths:
            board = image_to_chessboard(path)
            self.assertEqual(initial_chessboard, board, f"Chessboard piece placement is different than expected for '{path}'")

if __name__ == '__main__':
    unittest.main()
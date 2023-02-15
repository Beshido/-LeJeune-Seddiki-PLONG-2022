import unittest
from src.board_reader import PiecesType, get_cases_coordinates, check_cases_content

def checkboard_test(chessboard: list, paths: list) -> bool:
    
    return True

class TestBoardReader(unittest.TestCase):
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
            coordinates = get_cases_coordinates(path)
            board = check_cases_content(path, coordinates)
            for i in range(len(board)):
                for j in range(len(board[i])):
                    self.assertEqual(initial_chessboard[i][j], board[i][j].filled)
        
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
            coordinates = get_cases_coordinates(path)
            board = check_cases_content(path, coordinates)
            for i in range(len(board)):
                for j in range(len(board[i])):
                    self.assertEqual(initial_chessboard[i][j], board[i][j].filled)

if __name__ == '__main__':
    unittest.main()
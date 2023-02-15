import unittest
from src.board_reader import read_chessboard, PiecesType

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
            "img/chessboard-topview/image1.jpg",
            # "img/chessboard-topview/image2.png",
            "img/chessboard-topview/image3.webp",
            # "img/chessboard-topview/image4.jpg"
        ]
        for path in paths:
            board = read_chessboard(path)
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
            "img/chessboard-topview/image5.jpg"
        ]
        for path in paths:
            board = read_chessboard(path)
            for i in range(len(board)):
                for j in range(len(board[i])):
                    self.assertEqual(initial_chessboard[i][j], board[i][j].filled)

if __name__ == '__main__':
    unittest.main()
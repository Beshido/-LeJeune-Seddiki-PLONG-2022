import cv2, unittest
from src.board_reader import read_chessboard, PiecesType

class TestBoardReader(unittest.TestCase):
    def test_image_bank(self):
        initial_chessboard = [
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
            [ PiecesType.ANY, PiecesType.ANY, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.NONE, PiecesType.ANY, PiecesType.ANY ],
        ]
        self.assertEqual(initial_chessboard, read_chessboard("image1.jpg"))
        self.assertEqual(initial_chessboard, read_chessboard("image2.png"))
        self.assertEqual(initial_chessboard, read_chessboard("image3.webp"))
        self.assertEqual(initial_chessboard, read_chessboard("image4.jpg"))

if __name__ == '__main__':
    unittest.main()
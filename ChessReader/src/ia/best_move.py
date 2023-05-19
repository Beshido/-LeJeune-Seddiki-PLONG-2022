import chess.engine
import pathlib
from chess import Board, Move

STOCKFISH_PATH = pathlib.Path("utils/", "stockfish_20011801_x64.exe")

def get_best_move(board: Board, search_time: float = 2.) -> Move:
    """Renvoie le meilleur coup possible pour l'état du chessboard actuel."""

    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    result = e.play(board, chess.engine.Limit(time=search_time))
    e.close()
    return result.move
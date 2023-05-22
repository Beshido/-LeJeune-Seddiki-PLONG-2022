import chess.engine
import pathlib
from chess import Board, Move

STOCKFISH_PATH = pathlib.Path("utils/", "stockfish_20011801_x64.exe")
"""Chemin vers l'exécutable de Stockfish."""

SEARCH_TIME = 2.
"""Temps de recherche de Stockfish pour trouver le meilleur coup."""

def _get_best_move_from_board(board: Board) -> Move:
    """Renvoie le meilleur coup possible pour l'état du chessboard actuel. Prends en entrée un objet chess.Board."""

    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    result = e.play(board, chess.engine.Limit(time=SEARCH_TIME))
    e.close()
    return result.move

def get_best_move_from_fen(fen: str) -> Move:
    """Renvoie le meilleur coup possible pour l'état du chessboard actuel. Prends en entrée un FEN."""

    board = Board(fen)
    return _get_best_move_from_board(board)
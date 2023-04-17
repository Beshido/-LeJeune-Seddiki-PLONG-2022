import chess.engine
import random

board = chess.Board()
for i in range(random.randint(10, 40)):
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
random_fen = board.fen()

new_board = chess.Board()
new_board.set_fen(random_fen)

engine = chess.engine.SimpleEngine.popen_uci("../stockfish-11-win/Windows/stockfish_20011801_x64.exe")
result = engine.play(new_board, chess.engine.Limit(time=2.0))
print(new_board)
print("Best move:", result.move)
engine.close()

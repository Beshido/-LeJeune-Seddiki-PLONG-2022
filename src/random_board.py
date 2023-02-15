import chess
import chess.engine
import random

board = chess.Board()
for i in range(random.randint(10, 30)):
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
if(stockfish.is_fen_valid(board.fen())):
    print("FEN is valid")
    random_fen = board.fen()
else:
    print("FEN is invalid")
    random_fen = stockfish.generate_fen()

new_board = chess.Board()
new_board.set_fen(random_fen)

engine = chess.engine.SimpleEngine.popen_uci("../stockfish-11-win/Windows/stockfish_20011801_x64.exe")
result = engine.play(new_board, chess.engine.Limit(time=2.0))
print(new_board)
print("Best move:", result.move)
engine.close()

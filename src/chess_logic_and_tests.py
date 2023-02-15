import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("../stockfish-11-win/Windows/stockfish_20011801_x64.exe")
board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(time=2.0))

print(board)
print("Score:", info["score"])
print("Best move:", info["pv"][0])

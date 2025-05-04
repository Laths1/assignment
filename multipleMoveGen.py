import chess
import chess.engine
from statistics import mode

# autoMarkerPath = r"C:\Users\lathi\Documents\SCHOOL\Honours\AI\assignment\stockfish\stockfish.exe"
autoMarkerPath = '/opt/stockfish/stockfish'
numOfInput = int(input())
boards = [chess.Board(input()) for _ in range(numOfInput)]
engine = chess.engine.SimpleEngine.popen_uci(autoMarkerPath, setpgrp=True)
opponentColor = [not board.turn for board in boards]
kingSquares = [board.king(color) for board, color in zip(boards, opponentColor)]

for board, kingSquare in zip(boards, kingSquares):
    if kingSquare is not None:
        attackers = board.attackers(board.turn, kingSquare)
        if attackers:
            move = chess.Move(next(iter(attackers)), kingSquare)
            # print(move.uci())
            engine.quit()
            exit()
  
plays = [engine.play(board, chess.engine.Limit(time=0.5)) for board in boards]
moves = sorted([play.move.uci() for play in plays])
print(mode(moves))
engine.quit()



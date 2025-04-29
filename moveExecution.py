import chess
if __name__ == "__main__":
    board = chess.Board(input())
    board.push(chess.Move.from_uci(input()))
    print(board.fen())
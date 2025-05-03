import chess
from reconchess.utilities import without_opponent_pieces, is_illegal_castle

def printMoves(moves):
  list = []
  for move in moves:
      list.append(move)
  list.sort()
  for i in list:
      print(i)

if __name__ == "__main__":
    fen = input().strip()
    board = chess.Board(fen)
    moves = [move.uci() for move in board.pseudo_legal_moves]
    moves.append('0000') #add the null move first

    for move in without_opponent_pieces(board).generate_castling_moves():
        if not is_illegal_castle(board, move):
            moves.append(move.uci())

    printMoves(moves)
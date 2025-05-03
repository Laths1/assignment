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
    board = chess.Board('8/5k2/8/8/8/p1p1p2n/P1P1P3/RB2K2R w K - 12 45')
    moves = [move.uci() for move in board.pseudo_legal_moves]
    moves.append('0000') #add the null move first

    for move in without_opponent_pieces(board).generate_castling_moves():
        if not is_illegal_castle(board, move):
            moves.append(move.uci())

    printMoves(moves)
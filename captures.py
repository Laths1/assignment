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
    board = chess.Board(input())
    capture = input()
    moves = [move.uci() for move in board.pseudo_legal_moves]
    moves.append('0000') #add the null move first

    for move in without_opponent_pieces(board).generate_castling_moves():
        if not is_illegal_castle(board, move):
            moves.append(move.uci())

    #printMoves(set(moves))
    states = set()
    
    for move in set(moves): 
        if move[2:] == capture:
            temp_board = board.copy()
            temp_board.push(chess.Move.from_uci(move)) 
            states.add(temp_board.fen())

    printMoves(states)


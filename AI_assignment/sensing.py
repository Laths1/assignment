import chess

def compareWindows(squares, pieces, board):
    for square, piece in zip(squares, pieces):  
        piece_type = board.piece_type_at(square)   
        if piece_type is not None:
            piece_symbol = chess.piece_symbol(piece_type)  
            if piece_symbol.lower() != piece.lower(): 
                return False  
    return True 
   
if __name__ == "__main__":
    numOfStates = int(input())
    states = [chess.Board(input().strip()) for _ in range(numOfStates)]
    window = input().split(';')
    squares = []
    for w in window:
        squares.append(chess.parse_square(w.split(':')[0]))

    pieces = []
    for w in window:
        pieces.append(w.split(':')[1])
        
    matching_fens = []
    for board in states:
        if compareWindows(squares, pieces, board):
            matching_fens.append(board.fen())

    matching_fens.sort()
    for fen in matching_fens:
        print(fen)



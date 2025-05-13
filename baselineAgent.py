import chess
import chess.engine
from reconchess.utilities import without_opponent_pieces, is_illegal_castle
from collections import Counter
import random
from reconchess import *

class MyAgent(Player):

    def __init__(self):
        self.enginePath = r"C:\Users\lathi\Documents\SCHOOL\Honours\AI\assignment\stockfish\stockfish.exe"
        self.engine = chess.engine.SimpleEngine.popen_uci(self.enginePath, setpgrp=True)
        
    def handle_game_start(self, color, board, opponent_name):
        self.board = board
        self.color = color
        self.opponent_name = opponent_name
        self.boards = set()
        self.capture_square = None
        self.initial_board = board.copy()


    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        if captured_my_piece:
            moves = [move.uci() for move in self.board.pseudo_legal_moves]
            moves.append('0000')
            for move in without_opponent_pieces(self.board).generate_castling_moves():
                if not is_illegal_castle(self.board, move):
                    moves.append(move.uci())
            for move in set(moves): 
                if move[2:] == capture_square:
                    temp_board = self.board.copy()
                    temp_board.push(chess.Move.from_uci(move)) 
                    self.boards.add(temp_board)
            self.board.remove_piece_at(capture_square)
        
    
    def choose_sense(self, sense_actions, move_actions, seconds_left):
        algebraic_squares = [chess.SQUARE_NAMES[sq] for sq in sense_actions]
        
        non_edge_squares = [
            sq_index for sq_index, sq_name in zip(sense_actions, algebraic_squares)
            if sq_name[0] not in ('a', 'h') and sq_name[1] not in ('1', '8')
        ]
        
        return random.choice(non_edge_squares if non_edge_squares else sense_actions)

    def handle_sense_result(self, sense_result):
        def compareWindows(squares, pieces, board):
            for square, piece in zip(squares, pieces):  
                piece_type = board.piece_type_at(square)   
                if piece_type is not None:
                    piece_symbol = chess.piece_symbol(piece_type)  
                    if piece_symbol.lower() != piece.lower(): 
                        return False  
            return True
        
        squares = []
        pieces = []
        for square, piece in sense_result:
            squares.append(square)
            pieces.append(piece)
            
        matching_fens = []
        for board in self.boards:
            if compareWindows(squares, pieces, board):
                matching_fens.append(board)

        matching_fens.sort()
        self.boards = set(matching_fens)

    def choose_move(self, move_actions, seconds_left):
        if len(self.boards) > 10000:
            self.boards = set(random.sample(self.boards, 10000))

        opponentColor = [not board.turn for board in self.boards]
        kingSquares = [board.king(color) for board, color in zip(self.boards, opponentColor)]

        for board, kingSquare in zip(self.boards, kingSquares):
            if kingSquare is not None:
                attackers = board.attackers(board.turn, kingSquare)
                if attackers:
                    move = chess.Move(next(iter(attackers)), kingSquare)             

        plays = []
        for board in self.boards:
            try:
                result = self.engine.play(board, chess.engine.Limit(time=(10/len(self.boards))))
                if result.move:
                    plays.append(result)
            except Exception as e:
                pass

        moves = sorted([play.move.uci() for play in plays if play.move])

        if moves:
            return chess.Move.from_uci(Counter(moves).most_common(1)[0][0])
        else:
            return random.choice(move_actions)

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                            captured_opponent_piece: bool, capture_square: Optional[chess.Square]):
        # Update the main board with the taken move
        if taken_move is not None:
            self.board.push(taken_move)

        if requested_move != taken_move and requested_move is not None:
            self.boards = {board for board in self.boards if not board.is_legal(requested_move)}

        # filter after move is applied to all states
        # new_boards = set()
        # for board in self.boards:
        #     if taken_move is None:
        #         if not any(board.generate_legal_moves()):
        #             new_boards.add(board.copy())
        #     else:
        #         if board.is_legal(taken_move):
        #             new_board = board.copy()
        #             new_board.push(taken_move)
        #             new_boards.add(new_board)
        # self.boards = new_boards
            




    def handle_game_end(self, winner_color, win_reason, game_history):
        self.engine.quit()

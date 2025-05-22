
import chess
import chess.engine
from reconchess.utilities import without_opponent_pieces, is_illegal_castle
from collections import Counter
import random
import math
from reconchess import *
from reconchess import Player


class ImprovedAgent(Player):

    def __init__(self):
        self.enginePath = r"C:\Users\lathi\Documents\SCHOOL\Honours\AI\assignment\stockfish\stockfish.exe"
        self.engine = chess.engine.SimpleEngine.popen_uci(self.enginePath, setpgrp=True)
        self.king_square_counts = Counter()
        self.opponent_color = None

    def handle_game_start(self, color, board, opponent_name):
        self.board = board
        self.color = color
        self.opponent_name = opponent_name
        self.boards = set()
        self.capture_square = None
        self.my_piece_captured_square = None
        self.moveCount = 0
        possibleBlackKing = ["a7","b7","c7","d7","e7","f7","g7","h7"]
        possibleWhiteKing = ["a2","b2","c2","d2","e2","f2","g2","h2"]
        centreSquares = ["c4","c5","d4","d5","e4","e5","f4","f5"]
        self.centreSquares = [chess.parse_square(sq) for sq in centreSquares]
        self.possibleBlackKing = [chess.parse_square(sq) for sq in possibleBlackKing]
        self.possibleWhiteKing = [chess.parse_square(sq) for sq in possibleWhiteKing]
        
    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        self.my_piece_captured_square = capture_square
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

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Optional[Square]:

        def in_check_any_board():
            for board in self.boards:
                king_square = board.king(self.color)
                if king_square is None:
                    continue
                attackers = board.attackers(not self.color, king_square)
                if attackers:
                    return True, king_square
            return False, None

        # Check mode: if we suspect we're in check, sense near our king
        in_check, king_square = in_check_any_board()
        if in_check and king_square in sense_actions:
            return king_square
        elif in_check:
            # Sense around the king if possible
            neighbors = [
                king_square + offset
                for offset in [-9, -8, -7, -1, 1, 7, 8, 9]
                if chess.square_mirror(king_square + offset) in sense_actions
            ]
            possible_squares = [sq for sq in neighbors if 0 <= sq < 64 and sq in sense_actions]
            if possible_squares:
                return random.choice(possible_squares)

        # Normal strategy resumes here
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        if self.moveCount < 5:
            self.moveCount += 1
            return random.choice(self.centreSquares)

        likely_king_squares = []
        for board in self.boards:
            opp_color = not board.turn
            king_sq = board.king(opp_color)
            if king_sq is not None:
                likely_king_squares.append(king_sq)

        if likely_king_squares:
            most_common_king_square = Counter(likely_king_squares).most_common(1)[0][0]
            if most_common_king_square in sense_actions:
                return most_common_king_square

        rand_val = random.random()
        if rand_val < 0.5:
            if self.color == chess.WHITE:
                return random.choice([sq for sq in self.possibleBlackKing if sq in sense_actions])
            else:
                return random.choice([sq for sq in self.possibleWhiteKing if sq in sense_actions])
        elif rand_val < 0.75:
            return random.choice([sq for sq in self.centreSquares if sq in sense_actions])

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

        evasive_moves = []
        found_check = False

        for board in self.boards:
            king_sq = board.king(self.color)
            if king_sq is None:
                continue

            # Check if our king is under attack
            if board.is_attacked_by(not self.color, king_sq):
                found_check = True

                for move in move_actions:
                    if move not in board.legal_moves:
                        continue
                    temp_board = board.copy()
                    try:
                        temp_board.push(move)
                        new_king_sq = temp_board.king(self.color)
                        if new_king_sq is not None and not temp_board.is_attacked_by(not self.color, new_king_sq):
                            evasive_moves.append(move)
                    except:
                        continue

        if found_check:
            print(f"[DEBUG] King might be in check. Found {len(evasive_moves)} evasive moves.")

        if evasive_moves:
            return random.choice(evasive_moves)

        # Fallback: Use Stockfish to choose best move
        plays = []
        for board in self.boards:
            try:
                result = self.engine.play(board, chess.engine.Limit(time=0.5))
                if result.move:
                    plays.append(result)
            except Exception:
                continue

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


    def handle_game_end(self, winner_color, win_reason, game_history):
        self.engine.quit()

    def save_game_pgn(self, filename="game.pgn"):
        with open(filename, "w") as f:
            f.write(str(self.game_history))
        print(f"Saved game to {filename}")


    

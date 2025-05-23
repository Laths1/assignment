import chess
import chess.engine
from reconchess.utilities import without_opponent_pieces, is_illegal_castle
from collections import Counter
import random
from reconchess import *

class improvedAgent(Player):

    def __init__(self):
        self.enginePath = r"C:\Users\lathi\Documents\SCHOOL\Honours\AI\assignment\stockfish\stockfish.exe"
        self.engine = chess.engine.SimpleEngine.popen_uci(self.enginePath, setpgrp=True)
        self.king_square_counts = Counter()
        self.opponent_color = None

    def handle_game_start(self, color, board, opponent_name):
        self.board = board
        self.color = color
        self.opponent_name = opponent_name
        self.boards = []
        self.boards.append(board)
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
        new_boards = []
        
        for board in self.boards:
            temp_board = board.copy()
            temp_board.turn = not self.color  # Opponent's turn
            
            # Generate all legal moves including null move
            moves = list(temp_board.generate_legal_moves())
            moves.append(chess.Move.null())
            
            for move in moves:
                board_copy = temp_board.copy()
                try:
                    board_copy.push(move)
                except:
                    continue
                
                # Case 1: Opponent captured our piece
                if captured_my_piece:
                    if move.to_square == capture_square:
                        new_boards.append(board_copy)
                
                # Case 2: No capture occurred
                else:
                    # Either null move or non-capture move
                    if move == chess.Move.null() or (not temp_board.is_capture(move)):
                        new_boards.append(board_copy)
        
        # Ensure we keep at least the original board if no moves match
        if not new_boards and self.boards:
            new_boards = [list(self.boards)[0].copy()]
        
        self.boards = new_boards
        if self.boards:
            self.board = self.boards[0].copy()
    
    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Optional[Square]:
        
        if self.my_piece_captured_square:
            print("recaptured a piece")
            return self.my_piece_captured_square
        
        if self.moveCount < 5:
            self.moveCount += 1
            print("openning")
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
                print("Most likely king square")
                return most_common_king_square

        rand_val = random.random()
        if rand_val < 0.5:
            if self.color == chess.WHITE:
                print("Possible black king square")
                return random.choice([sq for sq in self.possibleBlackKing if sq in sense_actions])
            else:
                print("Possible white king square")
                return random.choice([sq for sq in self.possibleWhiteKing if sq in sense_actions])
        elif rand_val < 0.75:
            print("centre squares")
            return random.choice([sq for sq in self.centreSquares if sq in sense_actions])

        algebraic_squares = [chess.SQUARE_NAMES[sq] for sq in sense_actions]
        non_edge_squares = [
            sq_index for sq_index, sq_name in zip(sense_actions, algebraic_squares)
            if sq_name[0] not in ('a', 'h') and sq_name[1] not in ('1', '8')
        ]
        print("Random sense")
        return random.choice(non_edge_squares if non_edge_squares else sense_actions)

    def handle_sense_result(self, sense_result):
        """
        Filters possible board states based on the sensing result.
        Only keeps boards that match all sensed squares exactly.
        """
        # Convert sense_result to a dictionary for easier lookup
        sense_data = {square: piece for square, piece in sense_result}
        
        matching_boards = []
        for board in self.boards:
            match = True
            for square, sensed_piece in sense_result:
                board_piece = board.piece_at(square)
                
                # Case 1: Both squares are empty
                if sensed_piece is None and board_piece is None:
                    continue
                    
                # Case 2: One square is empty but the other isn't
                if (sensed_piece is None) != (board_piece is None):
                    match = False
                    break
                    
                # Case 3: Compare piece type and color
                if (sensed_piece.piece_type != board_piece.piece_type or 
                    sensed_piece.color != board_piece.color):
                    match = False
                    break
                    
            if match:
                matching_boards.append(board.copy())
        
        # Always keep at least the current board if all get filtered out
        if not matching_boards and self.boards:
            matching_boards = [list(self.boards)[0].copy()]
        self.boards = matching_boards

    def choose_move(self, move_actions, seconds_left):
    # 1) Build a list of legal chess.Move objects
        if len(self.boards) > 10000:
            self.boards = random.sample(self.boards, 10000)
        legal_moves = []
        for m in move_actions:
            try:
                legal_moves.append(chess.Move.from_uci(str(m)))
            except ValueError:
                continue
        if not legal_moves:
            return None

        # 2) First pass: look for any direct capture of the opponent’s king
        king_capture_moves = []
        for board in self.boards:
            # Whose king are we attacking?
            opp_color = not board.turn
            king_sq = board.king(opp_color)
            if king_sq is None:
                continue

            # Who attacks that square?
            attackers = board.attackers(board.turn, king_sq)
            if not attackers:
                continue

            # Pick one attacker → capture move
            frm = next(iter(attackers))
            cap_move = chess.Move(frm, king_sq)
            if cap_move in legal_moves:
                king_capture_moves.append(cap_move)

        if king_capture_moves:
            # If multiple boards allow different king‐captures, pick the most common
            return Counter(king_capture_moves).most_common(1)[0][0]

        # 3) Otherwise: poll Stockfish on each board and vote
        suggestions = []
        if len(self.boards) > 0:
            time_per_board = 10/len(self.boards)
        else:
            time_per_board = 0.2

        for board in self.boards:
            # rebuild from FEN to clear nulls/history
            temp = chess.Board(board.fen())
            temp.turn = board.turn

            # only legal moves
            roots = [m for m in temp.legal_moves if m in legal_moves]
            if not roots:
                continue

            try:
                result = self.engine.play(
                    temp,
                    chess.engine.Limit(time=time_per_board),
                    root_moves=roots
                )
                if result.move in legal_moves:
                    suggestions.append(result.move)
            except chess.engine.EngineError:
                continue

        if suggestions:
            return Counter(suggestions).most_common(1)[0][0]

        # 4) Fallback to random legal move
        return random.choice(legal_moves)



    def handle_move_result(self,
                       requested_move: Optional[chess.Move],
                       taken_move: Optional[chess.Move],
                       captured_opponent_piece: bool,
                       capture_square: Optional[chess.Square]):
    # 1) First: update your “true” board with whatever was actually played
        if taken_move is not None:
            if taken_move in self.board.legal_moves:
                self.board.push(taken_move)
            else:
                print(f"!!! Illegal on self.board: {taken_move.uci()} in {self.board.fen()}")

        # 2) Now update all your belief‐boards
        new_boards = []
        for b in self.boards:
            b2 = b.copy()
            b2.turn = self.color  # next turn is yours
            if taken_move in b2.legal_moves:
                try:
                    b2.push(taken_move)
                    new_boards.append(b2)
                except:
                    continue

        # 3) Fallback if all boards got invalidated
        if new_boards:
            self.boards = new_boards
        else:
            print("All belief‐boards invalidated—resetting to true board.")
            self.boards = [self.board.copy()]

        # 4) Finally, if you requested a move that wasn’t actually played,
        #    drop any boards where that requested move *is* legal
        if requested_move is not None and requested_move != taken_move:
            self.boards = [
                b for b in self.boards
                if requested_move not in b.legal_moves
            ]



    def handle_game_end(self, winner_color, win_reason, game_history):
        print(win_reason)
        self.engine.quit()

    

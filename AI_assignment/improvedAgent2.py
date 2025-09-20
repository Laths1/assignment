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
        self.my_piece_captured_square = None

    def handle_game_start(self, color, board, opponent_name):
        self.board = board
        self.color = color
        self.opponent_name = opponent_name
        self.boards = []
        self.boards.append(board)
        self.capture_square = None
        self.my_piece_captured_square = None
        self.moveCount = 0
        possibleBlackKing = ["b7","c7","d7","e7","f7","g7"]
        possibleWhiteKing = ["b2","c2","d2","e2","f2","g2"]
        centreSquares = ["c4","c5","d4","d5","e4","e5","f4","f5"]
        self.centreSquares = [chess.parse_square(sq) for sq in centreSquares]
        self.possibleBlackKing = [chess.parse_square(sq) for sq in possibleBlackKing]
        self.possibleWhiteKing = [chess.parse_square(sq) for sq in possibleWhiteKing]
        
    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)
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
            return self.my_piece_captured_square
        
        future_move = self.choose_move(move_actions, seconds_left)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            return future_move.to_square

        if self.moveCount < 5:
            self.moveCount += 1
            return random.choice(self.centreSquares)

        likely_king_squares = []
        for board in self.boards:
            opp_color = not board.turn
            king_sq = board.king(opp_color)
            if king_sq is not None:
                likely_king_squares.append(king_sq)

        rand_val = random.random()
        if likely_king_squares and rand_val < 0.6:
            most_common_king_square = Counter(likely_king_squares).most_common(1)[0][0]
            if most_common_king_square in sense_actions:
                return most_common_king_square
        if rand_val < 0.3:
            if self.color == chess.WHITE:
                return random.choice([sq for sq in self.possibleBlackKing if sq in sense_actions])
            else:
                return random.choice([sq for sq in self.possibleWhiteKing if sq in sense_actions])
        elif rand_val < 1:
            return random.choice([sq for sq in self.centreSquares if sq in sense_actions])

        algebraic_squares = [chess.SQUARE_NAMES[sq] for sq in sense_actions]
        non_edge_squares = [
            sq_index for sq_index, sq_name in zip(sense_actions, algebraic_squares)
            if sq_name[0] not in ('a', 'h') and sq_name[1] not in ('1', '8')
        ]
        return random.choice(non_edge_squares if non_edge_squares else sense_actions)

    def handle_sense_result(self, sense_result):
        """
        Filters possible board states based on the sensing result.
        Only keeps boards that match all sensed squares exactly.
        """
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)
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

    def _restart_engine(self):
        try:
            self.engine.quit()
        except:
            pass  # already dead or terminated
        print("Restarting Stockfish engine...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.enginePath, setpgrp=True)

    def choose_move(self, move_actions, seconds_left):
    # 1) Build a list of legal chess.Move objects
        if len(self.boards) > 10000:
            self.boards = random.sample(self.boards, 1000)
        legal_moves = []
        for m in move_actions:
            try:
                legal_moves.append(chess.Move.from_uci(str(m)))
            except ValueError:
                continue
        if not legal_moves:
            return None

        
      
        # 2) First pass: look for any direct capture of the opponent’s king
        enemy_king_square = self.board.king(not self.color)

        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                print("king capture")
                return chess.Move(attacker_square, enemy_king_square)

        # 3) Otherwise: poll Stockfish on each board and vote
        suggestions = []
        if len(self.boards) > 20:
            time_per_board = 10/len(self.boards)
        else:
            time_per_board = 0.005
        
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
            except chess.engine.EngineTerminatedError:
                # print("Stockfish crashed, attempting recovery...")
                self._restart_engine()
                continue
            except Exception as e:
                # print(f"Error during Stockfish evaluation: {e}")
                continue

        if suggestions:
            if seconds_left <10:
                print("random engine")
                return random.choice(suggestions)
            move = Counter(suggestions).most_common(1)[0][0]
            if move in legal_moves:
                print("engine")
                return move
        # 4) Fallback to random legal move
        print("random")
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

    

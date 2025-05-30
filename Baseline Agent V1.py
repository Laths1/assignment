import chess
import os
import chess.engine
import random
from reconchess import Player, Color, Square, GameHistory, WinReason
from reconchess.utilities import without_opponent_pieces, is_illegal_castle
from statistics import mode, StatisticsError
from typing import List, Tuple, Optional

class Agent(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.narrowed_states = None
        self.engine_path = r"C:\Users\fzm1209\Documents\stockfish.exe"
        self.engine = None  # store engine for reuse and cleanup


    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        if self.color == chess.WHITE:
            self.handle_opponent_move_result(False, None)

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece and capture_square is not None:
            moves = [move.uci() for move in self.board.pseudo_legal_moves]
            moves.append('0000')

            for move in without_opponent_pieces(self.board).generate_castling_moves():
                if not is_illegal_castle(self.board, move):
                    moves.append(move.uci())

            states = set()

            for move in set(moves): 
                if move[2:] == capture_square:
                    temp_board = self.board.copy()
                    temp_board.push(chess.Move.from_uci(move)) 
                    states.add(temp_board.fen())

            self.narrowed_states = states

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float):
        non_edge_squares = [
                            square for square in sense_actions
                            if chess.square_file(square) not in (0, 7) and chess.square_rank(square) not in (0, 7)
                            ]
        if not non_edge_squares:
            return random.choice(sense_actions)

        return random.choice(non_edge_squares)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)        
            
    def choose_move(self, move_actions, seconds_left):
        boards = [chess.Board(fen) for fen in self.narrowed_states] if self.narrowed_states else [self.board.copy()]
        if len(boards) > 10000:
            boards = random.sample(boards, 10000)

        time_limit = 10 / len(boards) if boards else 0.1

        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path, setpgrp=True)

        
        opponentColor = [not board.turn for board in boards]
        kingSquares = [board.king(color) for board, color in zip(boards, opponentColor)]

        for board, kingSquare in zip(boards, kingSquares):
            if kingSquare is not None:
                attackers = board.attackers(board.turn, kingSquare)
                if attackers:
                    move = chess.Move(next(iter(attackers)), kingSquare)
                    if move in move_actions:
                        return move
    
        plays = [self.engine.play(board, chess.engine.Limit(time=time_limit)) for board in boards]

        for board in boards:
            try:
                result = self.engine.play(board, chess.engine.Limit(time=0.5))
                if result.move in move_actions:
                    plays.append(result.move.uci())
            except Exception as e:
                continue
        moves = sorted([play.move.uci() for play in plays])

        if not moves:
            return random.choice(move_actions) if move_actions else None

        try:
            majority_move = mode(moves)
        except StatisticsError:
            majority_move = sorted(moves)[0]

        final_move = chess.Move.from_uci(majority_move)
        if final_move in move_actions:
            return final_move
        else:
            return random.choice(move_actions)

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                            captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None

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
        self.engine_path = r""

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        if self.color == 'white':
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

    def choose_sense(self, sense_actions, move_actions, seconds_left):
        non_edge_squares = [
                            square for square in sense_actions
                            if square[0] not in ('a', 'h') and square[1] not in ('1', '8')
                            ]
        if not non_edge_squares:
            return random.choice(sense_actions)

        return random.choice(non_edge_squares)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)        
            
    def choose_move(self, move_actions, seconds_left):
        boards = [chess.Board(fen) for fen in self.narrowed_states] if self.narrowed_states else [self.board.copy()]
        if len(boards) > 10000:
            boards = random.sample(boards, 10000)

        time_limit = 10 / len(boards) if boards else 0.1
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path, setpgrp=True)

        opponentColor = [not board.turn for board in boards]
        kingSquares = [board.king(color) for board, color in zip(boards, opponentColor)]

        for board, kingSquare in zip(boards, kingSquares):
            if kingSquare is not None:
                attackers = board.attackers(board.turn, kingSquare)
                if attackers:
                    move = chess.Move(next(iter(attackers)), kingSquare)
                    # print(move.uci())
                    engine.quit()
                    exit()

        plays = [engine.play(board, chess.engine.Limit(time=0.1)) for board in boards]
        moves = sorted([play.move.uci() for play in plays])

        try:
            majority_move = mode(moves)
        except StatisticsError:
            majority_move = sorted(moves)[0]

        engine.quit()
        return majority_move

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                            captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
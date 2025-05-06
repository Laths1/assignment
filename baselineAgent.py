import chess
import chess.engine

class MyAgent(Player):

    def __init__(self):
        pass

    def handle_game_start(self, color, board, opponent_name):
        pass

    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        pass

    def choose_sense(self, sense_actions, move_actions, seconds_left):
        pass

    def handle_sense_result(self, sense_result):
        pass

    def choose_move(self, move_actions, seconds_left):
        pass

    def handle_move_result(self, requested_move, taken_move, captured_opponent_piece, capture_square):
        pass

    def handle_game_end(self, winner_color, win_reason, game_history):
        pass

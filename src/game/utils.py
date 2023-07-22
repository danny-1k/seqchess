import chess.pgn

def board_to_pgn(board):
    game = str(chess.pgn.Game.from_board(board))
    return game
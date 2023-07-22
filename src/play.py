import torch
from vocab import Vocab
from game.agent import Agent
from game.utils import board_to_pgn
from models import GRU
from utils.utils import read_yaml
import chess
from vocab import Vocab
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--temp", type=float, default=.2, help="Temperature. 0 meaning greedy sampling and 1 meaning uniform sampling")

args = parser.parse_args()

temperature = args.temp
vocab = Vocab()

checkpoint = "../checkpoints/gru_20230719_090016_829091_ip-172-31-33-178/checkpoint.pt"
checkpoint = torch.load(checkpoint)
config = checkpoint["config"]

model = GRU(
    vocab_size=vocab.vocab_size,
    embed_size=config["EMBED_SIZE"],
    hidden_size=config["HIDDEN_SIZE"], 
    num_layers=config["NUM_LAYERS"], 
    dropout=config["DROPOUT"],
    device="cuda"
)
model.load_state_dict(checkpoint["model_state_dict"])

board = chess.Board()

agent = Agent(
    model=model,
    vocab=vocab,
    device="cuda"
)


def get_human_move(board):
    move = input("Your move (white) _> ")
    
    try:
        board.push_san(move)
        return

    except:
        print("Invalid move")
        get_human_move(board)


while not board.is_game_over():
    print(board)

    get_human_move(board)

    ai_move = agent.make_move(
        board,
        temperature=temperature
    )

    print(f"AI move (black) _> {ai_move}")
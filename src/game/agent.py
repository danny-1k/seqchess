import torch
import chess
from .utils import board_to_pgn
from tqdm import tqdm


class Agent:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def make_move(self, board, temperature=1, prompt=None):
        with torch.no_grad():
            pgn = board_to_pgn(board)
            
            if prompt: # prompting it to do a move in the favor of black / white
                if prompt in ["1/2-1/2", "1-0", "0-1"]:
                    pgn = pgn.replace("*", prompt)

            tokens, _ = self.vocab.encode(pgn)
            tokens = torch.Tensor(tokens).unsqueeze(0).to(self.model.device).long()
            tokens = tokens.to(self.device)

            p = (self.model(tokens).squeeze() / temperature).exp()
            p = torch.multinomial(p, 100) # sample top tokens that a legal move is likely to be in

            p = self.vocab.decode(p.tolist())
            print(p)

            for move in tqdm(p):
                try:
                    board.push_san(move)
                    break

                except: # invalid move
                    continue

            return str(move)
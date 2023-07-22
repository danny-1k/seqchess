import os
import chess.pgn
from tqdm import tqdm


class GameExtractor:
    def __init__(self, raw_dir, save_dir, max_num):
        self.raw_dir = raw_dir
        self.save_dir = save_dir
        self.max_num = max_num

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.exists(self.raw_dir):
            raise ValueError("`raw dir` does not exist")

    def read_game(self, f):
        pgn = open(f, "r")
        game = chess.pgn.read_game(pgn)
        
        return {"pgn": pgn, "game": game}

    def extract_games(self, pgn, game):
        while game:
            try:
                white = "".join([l for l in game.headers["White"] if l.isalpha() or l.isalnum()])
                black =  "".join([l for l in game.headers["Black"] if l.isalpha() or l.isalnum()])

                name = f"{white}_vs_{black}.pgn"

                if name in os.listdir(self.save_dir):
                    continue

                out = open(os.path.join(self.save_dir, name), "w")
                exporter = chess.pgn.FileExporter(out)
                
                game.accept(exporter)

                game = chess.pgn.read_game(pgn)

            except KeyboardInterrupt:
                print("Stopped extraction...")
                break

            except Exception as e:
                print(f"An error occured : ",e)
                continue
        pgn.close()

    def run(self, delete_raw):
        files = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir)]
        for f in tqdm(files):
                read = self.read_game(f)
                pgn, game = read["pgn"], read["game"]
                self.extract_games(pgn, game)

                if delete_raw:
                    os.remove(f)        


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--raw_dir", type=str, default="../data/raw")
    parser.add_argument("--save_dir", type=str, default="../data/splitted")
    parser.add_argument("--max_num", type=int, default=60_000)
    parser.add_argument("--delete_raw", type=bool, default=True)

    args = parser.parse_args()

    game_extractor = GameExtractor(
        raw_dir=args.raw_dir, save_dir=args.save_dir, max_num=args.max_num)

    game_extractor.run(delete_raw=args.delete_raw)
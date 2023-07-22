class Vocab:
    def __init__(self):
        board_numbers = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8'
        ]

        board_letters = [
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
        ]

        pieces = [
            "R",
            "N",
            "B",
            "K",
            "Q",
            "P",

            "r",
            "n",
            "b",
            "k",
            "q",
            "p"

        ]

        cell_moves = []

        for board_letter in board_letters:
            for board_number in board_numbers:
                cell_moves.append(board_letter + board_number)

        capture_moves = []
        explicit_moves = []

        for piece in pieces:
            # print(piece)
            for cell in cell_moves:
                capture_moves.append(f"{cell}+")
                capture_moves.append(f"{cell}#")


                capture_moves.append(f"{piece}x{cell}")
                capture_moves.append(f"{piece}x{cell}#") # capture moves that result in checkmate

                explicit_moves.append(f"{piece}{cell}")
                explicit_moves.append(f"{piece}{cell}#")


                for target_cell in cell_moves:

                    move = f"{piece}{cell[0]}x{target_cell}"
                    if move not in capture_moves:
                        capture_moves.append(move)
                        capture_moves.append(move+"+")
                        capture_moves.append(move+"#")

                    
                    move = f"{piece}{cell[1]}x{target_cell}"
                    if move not in capture_moves:
                        capture_moves.append(move)
                        capture_moves.append(move+"+")
                        capture_moves.append(move+"#")

                    move = f"{piece}{cell[0]}x{target_cell}"
                    if move not in capture_moves:
                        capture_moves.append(move)
                        capture_moves.append(move+"+")
                        capture_moves.append(move+"#")


                    explicit_move = f"{piece}{cell[0]}{target_cell}"

                    if explicit_move not in explicit_moves:
                        explicit_moves.append(explicit_move)
                        explicit_moves.append(explicit_move+"+")
                        explicit_moves.append(explicit_move+"#")


                    explicit_move_with_letter = f"{cell[0]}x{target_cell}"

                    if explicit_move_with_letter not in explicit_moves:
                        # print(explicit_move_with_letter)
                        explicit_moves.append(explicit_move_with_letter)
                        explicit_moves.append(explicit_move_with_letter+"+")
                        explicit_moves.append(explicit_move_with_letter+"#")


                    explicit_move_with_number = f"{piece}{cell[1]}{target_cell}"
                    if explicit_move_with_number not in explicit_moves:
                        explicit_moves.append(explicit_move_with_number)
                        explicit_moves.append(explicit_move_with_number+"+")
                        explicit_moves.append(explicit_move_with_number+"#")


        capture_moves_check = [f"{move}+" for move in capture_moves]
        cell_names_check = [f"{move}+" for move in cell_moves]
        explicit_moves_check = [f"{move}+" for move in explicit_moves]
        promotions = []

        for col in board_letters:
            for end in ["1", "8"]:
                for piece in pieces:
                    if piece not in "KkpP":
                        promotions.append(f"{col}{end}={piece}") #promotion
                        # it is also possible to reach a check with this move
                        promotions.append(f"{col}{end}={piece}+") #promotion
                        promotions.append(f"{col}{end}={piece}#") #promotion that leads to check

                        # not too explicit promotions that leads in a check
                        for board_letter in board_letters:
                            for board_number in board_numbers:
                                promotions.append(f"{col}x{board_letter}{board_number}={piece}")
                                promotions.append(f"{col}x{board_letter}{board_number}={piece}+")
                                promotions.append(f"{col}x{board_letter}{board_number}={piece}#")


        special_moves = [
            "O-O",  # castling
            "O-O+",  # castlng check
            "O-O#",  # the fuck happens and theres a checkmate from castling
            "O-O-O",  # queen-side castling
            "O-O-O+", # queen-side castling check
            "O-O-O#", # the fuck happens and theres a checkmate from queen-side castling
        ]

        special_tokens = [
            "SOS",
            "PAD",
            "1/2-1/2",
            "1-0",
            "0-1",
            "*" # unknown winner ???? bruh i dont wanna have to deal with this shit
        ]

        self.tokens = special_tokens + cell_moves + capture_moves + capture_moves_check + \
            cell_names_check + special_moves + explicit_moves + explicit_moves_check + promotions
        self.tokens = sorted(list(set(self.tokens)))

        self.token_ix = {token: ix for ix, token in enumerate(self.tokens)}
        self.ix_token = {ix: token for ix, token in enumerate(self.tokens)}

        self.vocab_size = len(self.tokens)

    def encode(self, pgn):
        pgn = pgn.replace("\n", " ")
        pgn = pgn.split("] ")[-1]
        tokens = [token for token in pgn.split(
            " ") if '.' not in token and token != '']
        result = tokens[-1]

        tokens = [self.token_ix[token] for token in tokens]

        return tokens, result

    def pad(self, seq, length):

        if len(seq) > length:
            raise ValueError(f"input sequence is longer than maximum length set {len(seq)} > {length}")

        num_pad = length-len(seq)

        seq += [self.token_ix["PAD"]]*num_pad

        return seq

    def decode(self, seq):
        return [self.ix_token[i] for i in seq]


if __name__ == "__main__":
    vocab = Vocab()

    pgn = open("../data/spli2tted/Zukovskij_vs_AlekhineAlexander.pgn", "r").read()
    pgn = pgn.split("\n\n")[1]  # remove game metadata
    tokens, _ = vocab.encode(pgn)

    vocab.pad(tokens, 100)

    print(tokens)

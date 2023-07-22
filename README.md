In this project, I test the idea of using sequence models to do next token prediction directly on the pgn representation of a game.

Can deep understanding of the game be achieved purely from this representation?

there are many ways the pgn could've been tokenized for sequence modelling (concise san notation etc)

but as I was too lazy, The vocabulary of this model contains a sufficiently large number of algebraic notation tokens (the vocab size was more than 100k tokens I belive).
This aspect can definitely be optimized in the future.

A bidirectional GRU and a transformer were trained on this dataset (the GRU performed significantly better)

At test time, the top 100 tokens are sampled from a multinomial distribution weighted by the probabilities from the network.
The top legal move is then returned.

One thing I found interesting was prompting the model to make a move that it learned to lead to a specific outcome. (by prompting with 1-0, 0-1, 1/2-1/2, *, etc the model's behaviour changes). I call this results injection.

## I just wann play

### Install dependencies
```
pip install -r requirements.txt
```

### Play
```
python play.py --temp [temperature value. defaults to 0.2] --prompt [results injection prompt (e.g 1/2-1/2, 1-0, 0-1, *)]
```
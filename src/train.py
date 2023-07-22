import os
import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from data import get_train_test_datasets
from vocab import Vocab
from models import GRU, Transformer
from utils.train import write_to_tb, accuracy
from utils.utils import get_hostname_and_time_string, read_yaml
from tqdm import tqdm


def run(args):
    SEED = args.seed
    DEVICE = args.device
    LOGDIR = args.logdir
    MODEL = args.model.lower()
    RUN_NAME = get_hostname_and_time_string()
    CHECKPOINT_DIR = f"../checkpoints/{MODEL}_{RUN_NAME}"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = read_yaml(args.config)

    vocab = Vocab()

    print(f"Vocab size : {vocab.vocab_size}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    if "gru" in MODEL:
        net = GRU(
            vocab_size=vocab.vocab_size,
            embed_size=config["EMBED_SIZE"],
            hidden_size=config["HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
            dropout=config["DROPOUT"],
            device=DEVICE
        )

    elif "transformer" in MODEL:
        net = Transformer(
            vocab_size=vocab.vocab_size,
            max_len=args.seq_len,
            num_layers=config["NUM_LAYERS"],
            n_heads=config["N_HEADS"],
            d_model=config["D_MODEL"],
            d_ff=config["D_FF"],
            dropout=config["DROPOUT"],
            device=DEVICE,
        )

    else:
        raise ValueError("Invalid model specified")


    optimizer = optim.Adam(net.parameters(), lr=config["LR"])
    lossfn = nn.CrossEntropyLoss()

    writer = SummaryWriter(
        log_dir=f"{LOGDIR}/{RUN_NAME}"
    )

    train, test = get_train_test_datasets(
        src="../data/splitted",
        train_pct=args.train_pct,
        seq_len=args.seq_len,
        vocab=vocab,
        device=DEVICE
    )


    best_loss = float("inf")
    best_accuracy = 0


    for epoch in tqdm(range(args.epochs)):

        train_loss_epoch = 0
        train_accuracy_epoch = 0

        test_loss_epoch = 0
        test_accuracy_epoch = 0

        net.train()
        for x, y in tqdm(train):
            optimizer.zero_grad()

            p = net(x)

            loss = lossfn(p, y)

            train_loss_epoch = .6*loss.item() + .4*train_loss_epoch

            train_accuracy_epoch = .6*accuracy(p.argmax(-1), y) + .4*train_accuracy_epoch

            loss.backward()

            optimizer.step()

        net.eval()
        with torch.no_grad():
            for x, y in tqdm(test):
                p = net(x)
                loss = lossfn(p, y)

                test_loss_epoch = .6*loss.item() + .4*test_loss_epoch

                test_accuracy_epoch = .6*accuracy(p.argmax(-1), y) + .4*test_accuracy_epoch

        write_to_tb(
            writer,
            epoch,
            net,
            scalars={
                "Loss/train": train_loss_epoch,
                "Loss/test": test_loss_epoch,
                "Accuracy/train": train_accuracy_epoch,
                "Accuracy/test": test_accuracy_epoch,
            }
        )


        if (test_accuracy_epoch > best_accuracy) or (test_loss_epoch < best_loss) :
            best_accuracy = test_accuracy_epoch
            best_loss = test_loss_epoch

            checkpoint = {
                "args": args,
                "config": config,
                "epoch": epoch+1,
                "max_epochs": args.epochs,
                "loss": best_loss,
                "accuracy": best_accuracy,
                "model_state_dict": net.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3740)
    parser.add_argument("--logdir", type=str, default="../logs")
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default="gru")
    parser.add_argument("--config", type=str, default="../configs/gru.yml")

    args = parser.parse_args()

    run(args)
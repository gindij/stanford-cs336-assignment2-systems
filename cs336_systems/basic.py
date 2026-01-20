import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def main(args: argparse.Namespace):
    assert args.profile_forward or args.profile_backward

    with open(args.configs_file, "r", encoding="utf-8") as configs_file:
        configs = json.load(configs_file)["configs"]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"

    rows = []
    for config in configs:
        print(f"profiling {config['name']}")
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=args.rope_theta,
        ).to(device)

        x = torch.randint(0, args.vocab_size - 1, (args.batch_size, args.context_length), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

        for _ in tqdm.tqdm(range(args.warmup_steps), desc="Warm-up"):
            yhat = model(x)
            loss = cross_entropy(yhat, y)
            loss.backward()

        forward_times = []
        backward_times = []
        for _ in tqdm.tqdm(range(args.profiling_steps), desc="Profiling"):
            forward_start = time.perf_counter()
            yhat = model(x)
            loss = cross_entropy(yhat, y)
            if args.profile_forward:
                forward_times.append(time.perf_counter() - forward_start)

            if args.profile_backward:
                backward_start = time.perf_counter()
                loss.backward()
                if device == "cuda:0":
                    torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - backward_start)

        for name, times in [("forward", forward_times), ("backward", backward_times)]:
            if len(times) > 0:
                max_time, min_time = float(np.max(times)), float(np.min(times))
                mean = float(np.mean(times))
                stddev = float(np.std(times))
                row = {
                    "size": config["name"],
                    "dir": name,
                    "max": max_time,
                    "min": min_time,
                    "mean": mean,
                    "std": stddev,
                }
                rows.append(row)

    pd.DataFrame(rows).to_csv(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--configs-file", type=str, default="benchmark_configs.json")
    parser.add_argument("--output-file", type=str, default="output.csv")

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--profiling-steps", type=int, default=1000)
    parser.add_argument("--profile-forward", action="store_true")
    parser.add_argument("--profile-backward", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    main(args)

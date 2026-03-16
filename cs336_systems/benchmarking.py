import argparse
import json
import time
import contextlib

import numpy as np
import pandas as pd
import torch
import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

import cs336_basics.model
from cs336_basics.model import annotated_scaled_dot_product_attention


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def main(args: argparse.Namespace):
    assert args.profile_forward or args.profile_backward

    with open(args.configs_file, "r", encoding="utf-8") as configs_file:
        configs = json.load(configs_file)["configs"]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"

    maybe_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    maybe_autocast = torch.autocast(device_type=device.split(":")[0]) if args.use_mixed_precision else contextlib.nullcontext()

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

        optimizer = AdamW(model.parameters())

        x = torch.randint(0, args.vocab_size - 1, (args.batch_size, args.context_length), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

        for _ in tqdm.tqdm(range(args.warmup_steps), desc="Warm-up"):
            with maybe_autocast:
                model.zero_grad()
                yhat = model(x)
                loss = cross_entropy(yhat, y)
                maybe_sync()
                loss.backward()
                maybe_sync()
                optimizer.step()
                maybe_sync()

        forward_times = []
        backward_times = []
        optimizer_times = []
        for _ in tqdm.tqdm(range(args.profiling_steps), desc="Profiling"):
            with maybe_autocast:
                model.zero_grad()
                forward_start = time.perf_counter()
                yhat = model(x)
                loss = cross_entropy(yhat, y)
                maybe_sync()
                if args.profile_forward:
                    forward_times.append(time.perf_counter() - forward_start)

                if args.profile_backward:
                    backward_start = time.perf_counter()
                    loss.backward()
                    maybe_sync()
                    backward_times.append(time.perf_counter() - backward_start)

                if args.profile_optimizer:
                    optimizer_start = time.perf_counter()
                    optimizer.step()
                    maybe_sync()
                    optimizer_times.append(time.perf_counter() - optimizer_start)

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

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profiling-steps", type=int, default=10)
    parser.add_argument("--profile-forward", action="store_true")
    parser.add_argument("--profile-backward", action="store_true")
    parser.add_argument("--profile-optimizer", action="store_true")
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    main(args)

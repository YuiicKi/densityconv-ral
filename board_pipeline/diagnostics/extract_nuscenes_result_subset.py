#!/usr/bin/env python3
"""Extract prediction entries matching the sample tokens in a smaller result JSON."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("token_source")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.token_source) as f:
        token_data = json.load(f)
    tokens = list(token_data["results"])

    with open(args.source) as f:
        source = json.load(f)
    missing = [token for token in tokens if token not in source["results"]]
    if missing:
        raise SystemExit(f"Missing tokens: {missing}")

    subset = {
        "meta": source["meta"],
        "results": {token: source["results"][token] for token in tokens},
    }
    with open(args.output, "w") as f:
        json.dump(subset, f, separators=(",", ":"))

    print(
        f"samples={len(tokens)} "
        f"detections={sum(len(v) for v in subset['results'].values())}"
    )


if __name__ == "__main__":
    main()

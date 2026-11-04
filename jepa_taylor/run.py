"""Train + evaluate one or more configurations.

  python run.py --configs direct piwsl jepa conservative --seeds 0 1 2
Results go to results/<config>_s<seed>.json, checkpoints to ckpt/.
"""
import argparse
import json
import os
import time
import torch

import lj
import evaluate as ev
from train import train

CONFIGS = {
    "direct":        dict(kind="direct"),
    "piwsl":         dict(kind="piwsl"),
    "jepa":          dict(kind="jepa"),
    "conservative":  dict(kind="conservative"),
    # ablations
    "jepa_noJEPA":   dict(kind="jepa", w_jepa=0.0),     # Taylor through an unanchored predictor
    "jepa_noTaylor": dict(kind="jepa", w_taylor=0.0),   # JEPA latent loss only
    "piwsl_fwd":     dict(kind="piwsl", taylor_mode="forward"),
    "jepa_fwd":      dict(kind="jepa", taylor_mode="forward"),
}

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(n_train, n_val=200):
    path = os.path.join(HERE, "data", "lj13.pt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(lj.make_dataset(), path)
    d = torch.load(path)
    sp = lambda a, b: {k: v[a:b] for k, v in d.items()}
    return {"train": {k: v.float() for k, v in sp(0, n_train).items()},
            "val": {k: v.float() for k, v in sp(2000, 2000 + n_val).items()},
            "test": sp(2000 + n_val, None)}                              # float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["direct", "piwsl", "jepa", "conservative"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--md_steps", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = load_data(a.n_train)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
    for cfg in a.configs:
        for seed in a.seeds:
            name = f"{cfg}{a.tag}_s{seed}"
            model, hist = train(data=data, seed=seed, epochs=a.epochs, **CONFIGS[cfg])
            torch.save(model.state_dict(), os.path.join(HERE, "ckpt", name + ".pt"))
            m = ev.f64(model)
            te = data["test"]
            t0 = time.time()
            res = {"config": cfg, "seed": seed, "n_train": a.n_train, "epochs": a.epochs,
                   "train_time_s": model.train_time}
            res.update(ev.accuracy(m, te))
            res.update(ev.curl(m, te))
            res.update(ev.latent_response(m, te))
            res.update(ev.timing(m, te))
            res["md"] = ev.md_suite(m, te, nsteps=a.md_steps)
            res["eval_time_s"] = time.time() - t0
            res["history"] = hist
            with open(os.path.join(HERE, "results", name + ".json"), "w") as fh:
                json.dump(res, fh, indent=1)
            print(name, json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()

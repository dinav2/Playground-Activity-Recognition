#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from src.playground_data.data_builder import build_window_mpgcn_input
from MPGCN.nets import MPGCN
from MPGCN.graphs import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-people", type=int, default=12)
    parser.add_argument("--reduct-ratio", type=int, default=16)
    parser.add_argument("--no-att", action="store_true", help="disable ST-Person attention")
    args = parser.parse_args()

    # 1) Build MP-GCN input from our .npz
    sample = build_window_mpgcn_input(
        args.npz,
        start_frame=args.start_frame,
        window_size=args.window_size,
        max_people=args.max_people,
    )

    x_joint     = sample["x_joint"]      # (1,1,2,T,V,M)
    y_scene     = sample["y_scene"]
    y_person    = sample["y_person"]     # (M,)
    person_mask = sample["person_mask"]  # (M,)
    scene_vocab = sample["scene_vocab"]
    meta        = sample["meta"]

    print("meta:", meta)
    print("x_joint shape:", x_joint.shape)
    print("y_scene:", y_scene, "->", scene_vocab[y_scene])
    print("y_person:", y_person)
    print("person_mask:", person_mask)

    # 2) Torch tensors
    x = torch.from_numpy(x_joint).float()   # (N=1, I=1, C=2, T,V,M)
    N, I, C, T, V, M = x.shape

    # 3) Graph for playground+COCO (single-person skeleton, multi-person handled by M dimension)
    g = Graph(
        dataset="playground",
        graph="coco",        # COCO-17 skeleton
        labeling="spatial",  # standard ST-GCN spatial partition
        part="body",         # use body-part partition from the coco branch
        max_hop=1,
    )
    A_np = g.A              # (K, V, V)
    parts = g.parts         # list of np.array, each is a body part

    print("A shape from Graph:", A_np.shape)
    print("parts:", parts)

    A = torch.from_numpy(A_np).float()

    # 4) Instantiate MPGCN (original architecture)
    num_input   = I
    num_channel = C
    data_shape  = (num_input, num_channel, T, V, M)
    num_class   = max(1, len(scene_vocab))

    use_att = False

    model = MPGCN(
        data_shape=data_shape,
        num_class=num_class,
        A=A,
        use_att=use_att,
        parts=parts,
        reduct_ratio=args.reduct_ratio,
    )

    model.eval()
    with torch.no_grad():
        logits, features = model(x)

    print("logits shape:", logits.shape)
    print("logits:", logits)


if __name__ == "__main__":
    main()


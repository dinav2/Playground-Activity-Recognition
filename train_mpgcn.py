#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F  # <- para BCE de consistencia

from MPGCN.nets import MPGCN
from MPGCN.graphs import Graph
from src.playground_data.data_builder import WindowDataset
from src.playground_data.labels import (
    SCENE_CLASSES,
    ACTION_CLASSES,
    ROLE_CLASSES,
    SAFETY_FLAG_CLASSES,
)

IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_npz_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*.npz"))


def build_model_from_sample(sample, use_att: bool, reduct_ratio: int) -> MPGCN:
    """
    Construye MPGCN tomando la shape del primer sample del dataset.
    sample["x"]: (I, C, T, V, M)
    """
    x = sample["x"]  # tensor
    I, C, T, V, M = x.shape

    g = Graph(
        dataset="playground",
        graph="coco_pano4",        # COCO-17 skeleton + 4 static objects
        labeling="spatial",
        part="body",
        max_hop=1,
    )
    A_np = g.A
    parts = g.parts
    A = torch.from_numpy(A_np).float()

    num_input = I
    num_channel = C
    data_shape = (num_input, num_channel, T, V, M)
    num_class = len(SCENE_CLASSES)

    model = MPGCN(
        data_shape=data_shape,
        num_class=num_class,
        A=A,
        use_att=use_att,
        parts=parts,
        reduct_ratio=reduct_ratio,
    )
    return model


def compute_scene_class_weights(dataset: WindowDataset, num_classes: int) -> torch.Tensor:
    """
    Calcula weights por clase de escena a partir del dataset de ventanas.
    weight[c] ~ 1 / freq(c).
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(dataset)):
        y_scene = int(dataset[i]["y_scene"])
        if 0 <= y_scene < num_classes:
            counts[y_scene] += 1

    total = counts.sum()
    counts = np.maximum(counts, 1)  # evitar cero

    # Peso inversamente proporcional a la frecuencia:
    weights = total / (num_classes * counts.astype(np.float32))
    weights = weights.astype(np.float32)  # <- forzar float32

    print("Scene class counts:", counts.tolist())
    print("Scene class weights:", weights.tolist())

    return torch.from_numpy(weights)  # float32


def flat_logits_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits: (B, M, C)
    labels: (B, M)
    -> logits_flat: (B*M, C), labels_flat: (B*M,), mask_valid: (B*M,) bool
    """
    B, M, _ = logits.shape
    logits_flat = logits.reshape(B * M, -1)
    labels_flat = labels.reshape(B * M)
    mask_valid = labels_flat != IGNORE_INDEX
    return logits_flat, labels_flat, mask_valid


def train_one_epoch(
    model,
    person_head_action,
    person_head_role,
    person_head_safety,
    loader,
    optimizer,
    criterion_scene,
    criterion_action,
    criterion_role,
    criterion_safety,
    device,
    use_person_heads: bool,
    lambda_action: float,
    lambda_role: float,
    lambda_safety: float,
    idx_play_object_normal: int | None,
    idx_play_object_risk: int | None,
    risk_indices: List[int],
    lambda_consistency: float,
):
    model.train()
    if use_person_heads:
        person_head_action.train()
        person_head_role.train()
        person_head_safety.train()

    total_loss = 0.0
    total_correct_scene = 0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)             # (B, I, C, T, V, M)
        y_scene = batch["y_scene"].to(device) # (B,)
        y_person = batch["y_person"].to(device)   # (B, M)
        y_role = batch["y_role"].to(device)       # (B, M)
        y_safety = batch["y_safety"].to(device)   # (B, M)

        optimizer.zero_grad()

        logits_scene, features = model(x)     # logits_scene: (B, num_scene)
                                             # features: (B, C_feat, T, V, M)
        B, C_feat, T, V, M = features.shape
        # pooling temporal + espacial → (B, M, C_feat)
        feat_pool = features.mean(dim=(2, 3)).permute(0, 2, 1)  # (B, M, C_feat)

        loss = criterion_scene(logits_scene, y_scene)

        if use_person_heads:
            # heads por persona
            logits_action = person_head_action(feat_pool)   # (B, M, A)
            logits_role   = person_head_role(feat_pool)     # (B, M, R)
            logits_safety = person_head_safety(feat_pool)   # (B, M, S)

            # Acción
            la_logits, la_labels, mask_a = flat_logits_labels(logits_action, y_person)
            if mask_a.any():
                loss_action = criterion_action(la_logits[mask_a], la_labels[mask_a])
                loss = loss + lambda_action * loss_action

            # Rol
            lr_logits, lr_labels, mask_r = flat_logits_labels(logits_role, y_role)
            if mask_r.any():
                loss_role = criterion_role(lr_logits[mask_r], lr_labels[mask_r])
                loss = loss + lambda_role * loss_role

            # Safety (per-person)
            ls_logits, ls_labels, mask_s = flat_logits_labels(logits_safety, y_safety)
            if mask_s.any():
                loss_safety = criterion_safety(ls_logits[mask_s], ls_labels[mask_s])
                loss = loss + lambda_safety * loss_safety

            # ---------- Consistency loss escena ↔ riesgo ----------
            if (
                idx_play_object_normal is not None
                and idx_play_object_risk is not None
                and len(risk_indices) > 0
            ):
                # probas de safety por persona
                probs_safety = logits_safety.softmax(dim=-1)  # (B, M, S)
                risk_probs_person = probs_safety[..., risk_indices].sum(dim=-1)  # (B, M)

                # prob de riesgo a nivel escena (max sobre personas)
                scene_risk_prob = risk_probs_person.max(dim=1).values  # (B,)

                # target binario: 1 si escena es play_object_risk, 0 si es play_object_normal
                is_scene_risky = torch.zeros_like(scene_risk_prob, dtype=torch.float32)
                is_scene_risky = is_scene_risky.to(device)

                mask_risk = (y_scene == idx_play_object_risk)
                mask_normal = (y_scene == idx_play_object_normal)

                is_scene_risky[mask_risk] = 1.0
                is_scene_risky[mask_normal] = 0.0

                # solo usamos ventanas de juego (normal o risk)
                mask_play = mask_risk | mask_normal
                if mask_play.any():
                    scene_risk_prob_play = scene_risk_prob[mask_play]
                    is_scene_risky_play = is_scene_risky[mask_play]

                    loss_consistency = F.binary_cross_entropy(
                        scene_risk_prob_play, is_scene_risky_play
                    )
                    loss = loss + lambda_consistency * loss_consistency
            # --------------------------------------

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        preds_scene = logits_scene.argmax(dim=1)
        total_correct_scene += (preds_scene == y_scene).sum().item()
        total_samples += B

    avg_loss = total_loss / max(1, total_samples)
    acc_scene = total_correct_scene / max(1, total_samples)
    return avg_loss, acc_scene


@torch.no_grad()
def eval_one_epoch(
    model,
    person_head_action,
    person_head_role,
    person_head_safety,
    loader,
    criterion_scene,
    criterion_action,
    criterion_role,
    criterion_safety,
    device,
    use_person_heads: bool,
    lambda_action: float,
    lambda_role: float,
    lambda_safety: float,
    idx_play_object_normal: int | None,
    idx_play_object_risk: int | None,
    risk_indices: List[int],
    lambda_consistency: float,
):
    model.eval()
    if use_person_heads:
        person_head_action.eval()
        person_head_role.eval()
        person_head_safety.eval()

    total_loss = 0.0
    total_correct_scene = 0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_scene = batch["y_scene"].to(device)
        y_person = batch["y_person"].to(device)
        y_role = batch["y_role"].to(device)
        y_safety = batch["y_safety"].to(device)

        logits_scene, features = model(x)
        B, C_feat, T, V, M = features.shape
        feat_pool = features.mean(dim=(2, 3)).permute(0, 2, 1)

        loss = criterion_scene(logits_scene, y_scene)

        if use_person_heads:
            logits_action = person_head_action(feat_pool)
            logits_role   = person_head_role(feat_pool)
            logits_safety = person_head_safety(feat_pool)

            la_logits, la_labels, mask_a = flat_logits_labels(logits_action, y_person)
            if mask_a.any():
                loss_action = criterion_action(la_logits[mask_a], la_labels[mask_a])
                loss = loss + lambda_action * loss_action

            lr_logits, lr_labels, mask_r = flat_logits_labels(logits_role, y_role)
            if mask_r.any():
                loss_role = criterion_role(lr_logits[mask_r], lr_labels[mask_r])
                loss = loss + lambda_role * loss_role

            ls_logits, ls_labels, mask_s = flat_logits_labels(logits_safety, y_safety)
            if mask_s.any():
                loss_safety = criterion_safety(ls_logits[mask_s], ls_labels[mask_s])
                loss = loss + lambda_safety * loss_safety

            # misma consistency para tener val_loss comparable
            if (
                idx_play_object_normal is not None
                and idx_play_object_risk is not None
                and len(risk_indices) > 0
            ):
                probs_safety = logits_safety.softmax(dim=-1)  # (B, M, S)
                risk_probs_person = probs_safety[..., risk_indices].sum(dim=-1)  # (B, M)
                scene_risk_prob = risk_probs_person.max(dim=1).values  # (B,)

                is_scene_risky = torch.zeros_like(scene_risk_prob, dtype=torch.float32)
                is_scene_risky = is_scene_risky.to(device)

                mask_risk = (y_scene == idx_play_object_risk)
                mask_normal = (y_scene == idx_play_object_normal)

                is_scene_risky[mask_risk] = 1.0
                is_scene_risky[mask_normal] = 0.0

                mask_play = mask_risk | mask_normal
                if mask_play.any():
                    scene_risk_prob_play = scene_risk_prob[mask_play]
                    is_scene_risky_play = is_scene_risky[mask_play]

                    loss_consistency = F.binary_cross_entropy(
                        scene_risk_prob_play, is_scene_risky_play
                    )
                    loss = loss + lambda_consistency * loss_consistency

        total_loss += loss.item() * B
        preds_scene = logits_scene.argmax(dim=1)
        total_correct_scene += (preds_scene == y_scene).sum().item()
        total_samples += B

    avg_loss = total_loss / max(1, total_samples)
    acc_scene = total_correct_scene / max(1, total_samples)
    return avg_loss, acc_scene


@torch.no_grad()
def eval_confusion_matrix(model, loader, device, num_classes: int):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for batch in loader:
        x = batch["x"].to(device)
        y_scene = batch["y_scene"].to(device)

        logits_scene, _ = model(x)
        preds = logits_scene.argmax(dim=1)  # (B,)

        for t, p in zip(y_scene.view(-1), preds.view(-1)):
            if 0 <= t < num_classes:
                cm[t.long(), p.long()] += 1

    return cm.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/npz"))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--max-people", type=int, default=12)
    parser.add_argument("--stride", type=int, default=15, help="sliding window stride")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-att", action="store_true", help="disable ST-Person attention")
    parser.add_argument("--reduct-ratio", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", type=Path, default=Path("mpgcn_playground.pt"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time-jitter", type=int, default=2)
    parser.add_argument("--coord-noise-std", type=float, default=2.0)
    parser.add_argument("--person-dropout-prob", type=float, default=0.1)


    parser.add_argument("--use-person-heads", action="store_true")
    parser.add_argument("--lambda-action", type=float, default=0.5)
    parser.add_argument("--lambda-role", type=float, default=0.25)
    parser.add_argument("--lambda-safety", type=float, default=0.25)
    parser.add_argument(
        "--lambda-consistency",
        type=float,
        default=0.5,
        help="peso de la loss de consistencia escena↔riesgo (solo play_object_*)",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npz_files = find_npz_files(args.data_dir)
    from src.playground_data.labels import SCENE_CLASSES  # now 3 classes

    # Keep only videos whose scene label is in our 3-class vocabulary
    KEEP_SCENES = set(SCENE_CLASSES)

    def filter_npz_by_scene(npz_paths):
        kept = []
        dropped = []
        for p in npz_paths:
            with np.load(p, allow_pickle=True) as d:
                # Each .npz has scene_vocab (array of strings) and scene_label (int index into that array)
                scene_vocab = d["scene_vocab"]
                scene_label = int(d["scene_label"])

                # scene_vocab is e.g. array(['adult_assisting'], dtype=object)
                scene_name = str(scene_vocab[scene_label])

            if scene_name in KEEP_SCENES:
                kept.append(p)
            else:
                dropped.append((p, scene_name))

        print(f"Filtered npz files: kept {len(kept)}, dropped {len(dropped)}")
        if dropped:
            print("Dropped scene labels:", sorted({s for _, s in dropped}))
        return kept

    npz_files = filter_npz_by_scene(npz_files)
    if not npz_files:
        raise SystemExit("No .npz files left after scene filtering!")


    print(f"Found {len(npz_files)} npz files")

    # ---------- split train / val por archivos ----------
    indices = list(range(len(npz_files)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1.0 - args.val_split))
    split_idx = max(1, min(split_idx, len(indices) - 1))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_files = [npz_files[i] for i in train_indices]
    val_files = [npz_files[i] for i in val_indices]

    print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}")

    # ---------- datasets ----------
    train_dataset = WindowDataset(
        npz_paths=train_files,
        window_size=args.window_size,
        max_people=args.max_people,
        stride=args.stride,
        ignore_index=IGNORE_INDEX,
        augment=args.augment,
        time_jitter=args.time_jitter,
        coord_noise_std=args.coord_noise_std,
        person_dropout_prob=args.person_dropout_prob,
    )

    val_dataset = WindowDataset(
        npz_paths=val_files,
        window_size=args.window_size,
        max_people=args.max_people,
        stride=args.stride,
        ignore_index=IGNORE_INDEX,
        augment=False,  # val siempre limpio
    )


    if len(train_dataset) == 0:
        raise SystemExit("Train WindowDataset is empty – check window_size/stride.")
    if len(val_dataset) == 0:
        print("WARNING: Val WindowDataset is empty – val metrics will be NaN.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------- modelo ----------
    first_sample = train_dataset[0]
    model = build_model_from_sample(
        first_sample,
        use_att=not args.no_att,
        reduct_ratio=args.reduct_ratio,
    )
    model.to(device)

    print(f"Num scene classes: {len(SCENE_CLASSES)} -> {SCENE_CLASSES}")

    num_action_classes = len(ACTION_CLASSES)
    num_role_classes = len(ROLE_CLASSES)
    num_safety_classes = len(SAFETY_FLAG_CLASSES)

    # person heads (aunque no se usen, es más simple tenerlos siempre)
    person_head_action = nn.Linear(256, num_action_classes).to(device)
    person_head_role   = nn.Linear(256, num_role_classes).to(device)
    person_head_safety = nn.Linear(256, num_safety_classes).to(device)

    # ---------- class weights para escenas ----------
    scene_weights = compute_scene_class_weights(
        train_dataset,
        num_classes=len(SCENE_CLASSES),
    ).to(device).float()

    criterion_scene  = nn.CrossEntropyLoss(weight=scene_weights)
    criterion_action = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_role   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_safety = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # parámetros del optimizador
    params = list(model.parameters())
    if args.use_person_heads:
        params += list(person_head_action.parameters())
        params += list(person_head_role.parameters())
        params += list(person_head_safety.parameters())

    optimizer = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"use_person_heads = {args.use_person_heads}")
    if args.use_person_heads:
        print(
            f"lambda_action={args.lambda_action}, "
            f"lambda_role={args.lambda_role}, "
            f"lambda_safety={args.lambda_safety}, "
            f"lambda_consistency={args.lambda_consistency}"
        )

    # índices para consistency loss
    try:
        idx_play_object_normal = SCENE_CLASSES.index("play_object_normal")
    except ValueError:
        idx_play_object_normal = None
    try:
        idx_play_object_risk = SCENE_CLASSES.index("play_object_risk")
    except ValueError:
        idx_play_object_risk = None

    risk_indices = [
        i for i, name in enumerate(SAFETY_FLAG_CLASSES)
        if isinstance(name, str) and name.startswith("risk_")
    ]
    
    best_val_acc_scene = -1.0
    best_epoch = 0   

    # ---------- training loop ----------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            person_head_action,
            person_head_role,
            person_head_safety,
            train_loader,
            optimizer,
            criterion_scene,
            criterion_action,
            criterion_role,
            criterion_safety,
            device,
            use_person_heads=args.use_person_heads,
            lambda_action=args.lambda_action,
            lambda_role=args.lambda_role,
            lambda_safety=args.lambda_safety,
            idx_play_object_normal=idx_play_object_normal,
            idx_play_object_risk=idx_play_object_risk,
            risk_indices=risk_indices,
            lambda_consistency=args.lambda_consistency,
        )

        val_loss, val_acc = eval_one_epoch(
            model,
            person_head_action,
            person_head_role,
            person_head_safety,
            val_loader,
            criterion_scene,
            criterion_action,
            criterion_role,
            criterion_safety,
            device,
            use_person_heads=args.use_person_heads,
            lambda_action=args.lambda_action,
            lambda_role=args.lambda_role,
            lambda_safety=args.lambda_safety,
            idx_play_object_normal=idx_play_object_normal,
            idx_play_object_risk=idx_play_object_risk,
            risk_indices=risk_indices,
            lambda_consistency=args.lambda_consistency,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}  train_acc_scene={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc_scene={val_acc:.4f}"
        )

        # --------- guardar mejor checkpoint ---------
        if val_acc > best_val_acc_scene:
            best_val_acc_scene = val_acc
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc_scene": best_val_acc_scene,
                "args": vars(args),
            }

            if args.use_person_heads:
                checkpoint["person_head_action_state_dict"] = person_head_action.state_dict()
                checkpoint["person_head_role_state_dict"] = person_head_role.state_dict()
                checkpoint["person_head_safety_state_dict"] = person_head_safety.state_dict()

            torch.save(checkpoint, args.save)
            print(
                f"  -> New best val_acc_scene={val_acc:.4f} at epoch {epoch}, "
                f"checkpoint saved to {args.save}"
            )  
    
    best_ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # ---------- confusion matrix en VAL ----------
    cm = eval_confusion_matrix(model, val_loader, device, num_classes=len(SCENE_CLASSES))
    print("\nFinal confusion matrix on VAL (rows=GT, cols=Pred):")
    print("Scene label index mapping:")
    for idx, name in enumerate(SCENE_CLASSES):
        print(f"  {idx}: {name}")
    print(cm.numpy())


if __name__ == "__main__":
    main()


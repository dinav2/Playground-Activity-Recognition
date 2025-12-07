#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_bar(counter: Counter, title: str, xlabel: str, ylabel: str, out_path: Path, rotation: int = 45):
    if not counter:
        print(f"[warn] No hay datos para {title}, no se genera gráfica.")
        return

    labels, values = zip(*sorted(counter.items(), key=lambda kv: kv[1], reverse=True))

    plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=rotation, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[ok] Guardada gráfica: {out_path}")


def plot_hist(values, title: str, xlabel: str, ylabel: str, out_path: Path):
    if not values:
        print(f"[warn] No hay datos para {title}, no se genera gráfica.")
        return
    max_val = max(values)
    bins = np.arange(0, max_val + 2) - 0.5  # bins centrados en enteros

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins)
    plt.xticks(range(0, max_val + 1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[ok] Guardada gráfica: {out_path}")


def analyze_npz_dir(npz_dir: Path, output_dir: Path):
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No se encontraron .npz en {npz_dir}")

    ensure_dir(output_dir)

    # Contadores globales
    scene_counts = Counter()        # escenas por video
    scene_frame_counts = Counter()  # escenas por frame
    people_per_frame = []           # lista con #personas por frame

    action_counts = Counter()
    role_counts = Counter()
    safety_counts = Counter()

    # acciones / roles / safety por escena
    actions_per_scene = defaultdict(Counter)
    roles_per_scene = defaultdict(Counter)
    safety_per_scene = defaultdict(Counter)

    total_frames = 0
    total_videos = 0

    print(f"[info] Analizando {len(npz_files)} archivos en {npz_dir}...")

    for path in npz_files:
        data = load_npz(path)

        keypoints = data["keypoints"]      # (T, N, J, 2)
        mask = data["mask"]                # (T, N)
        actions = data["actions"]          # (T, N)
        roles = data["roles"]              # (T, N)
        safety_flags = data["safety_flags"]  # (T, N)

        # Vocabularios (pueden ser arrays vacíos en algunos casos)
        scene_vocab = data["scene_vocab"]          # [scene_name] normalmente
        scene_label = int(data["scene_label"])     # índice en scene_vocab
        actions_vocab = list(data["actions_vocab"])
        roles_vocab = list(data["roles_vocab"])
        safety_vocab = list(data["safety_vocab"])

        T, N, _, _ = keypoints.shape
        total_frames += T
        total_videos += 1

        # Escena del video
        if len(scene_vocab) > 0 and scene_label >= 0:
            scene_name = str(scene_vocab[scene_label])
        else:
            scene_name = "unknown"

        scene_counts[scene_name] += 1
        scene_frame_counts[scene_name] += T

        # Número de personas por frame
        frame_person_counts = mask.sum(axis=1)  # (T,)
        people_per_frame.extend(frame_person_counts.tolist())

        # Acciones / roles / safety por persona
        # Solo consideramos personas presentes (mask==1) y labels >= 0
        for t in range(T):
            for n in range(N):
                if mask[t, n] == 0:
                    continue

                # Acción
                a_id = int(actions[t, n])
                if a_id >= 0 and a_id < len(actions_vocab):
                    a_name = str(actions_vocab[a_id])
                    action_counts[a_name] += 1
                    actions_per_scene[scene_name][a_name] += 1

                # Rol
                r_id = int(roles[t, n])
                if r_id >= 0 and r_id < len(roles_vocab):
                    r_name = str(roles_vocab[r_id])
                    role_counts[r_name] += 1
                    roles_per_scene[scene_name][r_name] += 1

                # Safety flag
                s_id = int(safety_flags[t, n])
                if s_id >= 0 and s_id < len(safety_vocab):
                    s_name = str(safety_vocab[s_id])
                    safety_counts[s_name] += 1
                    safety_per_scene[scene_name][s_name] += 1

    # ------------------ Imprimir resumen en consola ------------------
    print("\n===== RESUMEN GLOBAL =====")
    print(f"Total de videos:  {total_videos}")
    print(f"Total de frames:  {total_frames}")
    print(f"Promedio personas por frame: {np.mean(people_per_frame):.2f}")
    print("\nEscenas (por video):")
    for k, v in scene_counts.most_common():
        print(f"  {k:20s}: {v:4d}")

    print("\nEscenas (por frame):")
    for k, v in scene_frame_counts.most_common():
        print(f"  {k:20s}: {v:6d}")

    print("\nTop acciones globales:")
    for k, v in action_counts.most_common(10):
        print(f"  {k:20s}: {v:6d}")

    print("\nTop roles globales:")
    for k, v in role_counts.most_common(10):
        print(f"  {k:20s}: {v:6d}")

    print("\nTop safety flags globales:")
    for k, v in safety_counts.most_common(10):
        print(f"  {k:20s}: {v:6d}")

    # ------------------ Gráficas ------------------
    # 1) Escenas por video
    plot_bar(
        scene_counts,
        title="Escenas (conteo de videos)",
        xlabel="Escena",
        ylabel="# videos",
        out_path=output_dir / "scene_counts_videos.png",
    )

    # 2) Escenas por frame
    plot_bar(
        scene_frame_counts,
        title="Escenas (conteo de frames)",
        xlabel="Escena",
        ylabel="# frames",
        out_path=output_dir / "scene_counts_frames.png",
    )

    # 3) Histograma de personas por frame
    plot_hist(
        people_per_frame,
        title="Número de personas por frame",
        xlabel="# personas",
        ylabel="# frames",
        out_path=output_dir / "people_per_frame_hist.png",
    )

    # 4) Acciones globales
    plot_bar(
        action_counts,
        title="Acciones individuales (global)",
        xlabel="Acción",
        ylabel="# apariciones",
        out_path=output_dir / "actions_global.png",
    )

    # 5) Roles globales
    plot_bar(
        role_counts,
        title="Roles (global)",
        xlabel="Rol",
        ylabel="# apariciones",
        out_path=output_dir / "roles_global.png",
    )

    # 6) Safety flags globales
    plot_bar(
        safety_counts,
        title="Safety flags (global)",
        xlabel="Safety flag",
        ylabel="# apariciones",
        out_path=output_dir / "safety_global.png",
    )

    # 7) Acciones por escena (una gráfica por escena, para las más frecuentes)
    for scene_name, cnt in actions_per_scene.items():
        if not cnt:
            continue
        slug = scene_name.replace(" ", "_")
        plot_bar(
            cnt,
            title=f"Acciones en escena: {scene_name}",
            xlabel="Acción",
            ylabel="# apariciones",
            out_path=output_dir / f"actions_{slug}.png",
        )

    # 8) (Opcional) Roles por escena
    for scene_name, cnt in roles_per_scene.items():
        if not cnt:
            continue
        slug = scene_name.replace(" ", "_")
        plot_bar(
            cnt,
            title=f"Roles en escena: {scene_name}",
            xlabel="Rol",
            ylabel="# apariciones",
            out_path=output_dir / f"roles_{slug}.png",
        )

    # 9) (Opcional) Safety por escena
    for scene_name, cnt in safety_per_scene.items():
        if not cnt:
            continue
        slug = scene_name.replace(" ", "_")
        plot_bar(
            cnt,
            title=f"Safety flags en escena: {scene_name}",
            xlabel="Safety flag",
            ylabel="# apariciones",
            out_path=output_dir / f"safety_{slug}.png",
        )

    print(f"\n[done] Gráficas y resumen guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza estadísticas del dataset (escenas, #personas, acciones, roles, safety) a partir de .npz."
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path("data/npz"),
        help="Directorio con los .npz (por defecto: data/npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stats"),
        help="Directorio donde guardar las gráficas (por defecto: data/stats)",
    )
    args = parser.parse_args()

    analyze_npz_dir(args.npz_dir, args.output_dir)


if __name__ == "__main__":
    main()


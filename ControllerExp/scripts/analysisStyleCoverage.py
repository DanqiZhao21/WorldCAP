# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Style Coverage Analysis for Controller Styles
# ===========================================

# This script computes **controller style coverage metrics** from a pre-generated
# controller style bank (e.g. controller_styles.npz).

# It implements three levels of metrics:

# (A) Trajectory Behavior Diversity (pairwise distance)
# (B) Clustering-based Coverage (KMeans + silhouette)
# (C) Optional Latent Space Coverage (if controller embeddings are provided)

# The metrics are designed to be **paper-ready** (ECCV / NeurIPS style) and
# operate purely in **behavior space**, not parameter space.

# Author: WoTE / Controller-as-Latent Variable
# """

# import argparse
# import os
# import numpy as np
# from typing import Dict, Any

# from scipy.spatial.distance import pdist
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA

# # ------------------------------------------------------------
# # Descriptor Definition
# # ------------------------------------------------------------

# def traj_descriptor(exec_traj: np.ndarray) -> np.ndarray:
#     """
#     Compute a low-dimensional behavior descriptor for a controller style.

#     Args:
#         exec_traj: np.ndarray of shape [N_traj, T, 3]
#                    (x, y, yaw)

#     Returns:
#         descriptor: np.ndarray of shape [4]
#     """
#     assert exec_traj.ndim == 3 and exec_traj.shape[-1] == 3

#     xy = exec_traj[..., :2]
#     yaw = exec_traj[..., 2]

#     lateral_std = xy[..., 0].std()
#     longitudinal_std = xy[..., 1].std()

#     yaw_rate = np.abs(np.diff(yaw, axis=1)).mean()
#     total_heading_change = np.abs(yaw[:, -1] - yaw[:, 0]).mean()

#     return np.array([
#         lateral_std,
#         longitudinal_std,
#         yaw_rate,
#         total_heading_change,
#     ], dtype=np.float64)


# # ------------------------------------------------------------
# # Metric A: Pairwise Behavior Diversity
# # ------------------------------------------------------------

# def compute_pairwise_coverage(descriptors: np.ndarray) -> Dict[str, float]:
#     """
#     Compute pairwise L2 distance statistics between style descriptors.
#     """
#     pairwise_dist = pdist(descriptors, metric="euclidean")

#     return {
#         "pairwise_mean": float(pairwise_dist.mean()),
#         "pairwise_min": float(pairwise_dist.min()),
#         "pairwise_max": float(pairwise_dist.max()),
#     }


# # ------------------------------------------------------------
# # Metric B: Clustering-based Coverage
# # ------------------------------------------------------------

# def compute_clustering_coverage(
#     descriptors: np.ndarray,
#     k_list=(4, 6, 8, 10),
#     seed: int = 0,
# ) -> Dict[str, Any]:
#     """
#     Perform KMeans clustering and compute silhouette scores.
#     """
#     results = {}

#     for k in k_list:
#         if k >= len(descriptors):
#             continue

#         kmeans = KMeans(n_clusters=k, random_state=seed)
#         labels = kmeans.fit_predict(descriptors)
#         sil = silhouette_score(descriptors, labels)

#         results[f"k={k}"] = {
#             "silhouette": float(sil),
#             "inertia": float(kmeans.inertia_),
#         }

#     return results


# # ------------------------------------------------------------
# # Metric C: Latent Space Coverage (Optional)
# # ------------------------------------------------------------

# def compute_latent_coverage(latents: np.ndarray) -> Dict[str, Any]:
#     """
#     Analyze coverage of controller latent embeddings.
#     """
#     pca = PCA()
#     pca.fit(latents)

#     explained = np.cumsum(pca.explained_variance_ratio_)

#     return {
#         "latent_dim": latents.shape[1],
#         "pca_explained_cumsum": explained.tolist(),
#         "pc3_coverage": float(explained[min(2, len(explained)-1)]),
#         "pc5_coverage": float(explained[min(4, len(explained)-1)]),
#     }


# # ------------------------------------------------------------
# # Main
# # ------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser("Controller Style Coverage Analysis")
#     parser.add_argument(
#         "--style_npz",
#         type=str,
#         required=True,
#         help="Path to controller_styles.npz",
#     )
#     parser.add_argument(
#         "--latent_key",
#         type=str,
#         default=None,
#         help="Optional key in npz for controller latents (e.g. 'latents')",
#     )
#     parser.add_argument(
#         "--k_list",
#         type=int,
#         nargs="+",
#         default=[4, 6, 8, 10],
#     )
#     args = parser.parse_args()

#     assert os.path.exists(args.style_npz), f"File not found: {args.style_npz}"

#     data = np.load(args.style_npz, allow_pickle=True)

#     assert "exec_trajs" in data, "exec_trajs not found in npz"
#     exec_trajs = data["exec_trajs"]

#     print(f"[INFO] Loaded {len(exec_trajs)} controller styles")

#     # --------------------------------------------------------
#     # Descriptor extraction
#     # --------------------------------------------------------
#     descriptors = np.stack([
#         traj_descriptor(exec_trajs[i])
#         for i in range(len(exec_trajs))
#     ])

#     print("[INFO] Descriptor shape:", descriptors.shape)

#     # --------------------------------------------------------
#     # Metric A
#     # --------------------------------------------------------
#     pairwise_stats = compute_pairwise_coverage(descriptors)

#     # --------------------------------------------------------
#     # Metric B
#     # --------------------------------------------------------
#     cluster_stats = compute_clustering_coverage(
#         descriptors, k_list=args.k_list
#     )

#     # --------------------------------------------------------
#     # Metric C (Optional)
#     # --------------------------------------------------------
#     latent_stats = None
#     if args.latent_key is not None:
#         assert args.latent_key in data, f"{args.latent_key} not found in npz"
#         latents = data[args.latent_key]
#         latent_stats = compute_latent_coverage(latents)

#     # --------------------------------------------------------
#     # Print summary (paper-ready)
#     # --------------------------------------------------------
#     print("\n========== Style Coverage Summary ==========")

#     print("\n[Metric A] Pairwise Behavior Diversity")
#     for k, v in pairwise_stats.items():
#         print(f"  {k}: {v:.4f}")

#     print("\n[Metric B] Clustering-based Coverage")
#     for k, stats in cluster_stats.items():
#         print(f"  {k}: silhouette={stats['silhouette']:.3f}, inertia={stats['inertia']:.2f}")

#     if latent_stats is not None:
#         print("\n[Metric C] Latent Space Coverage")
#         print(f"  latent_dim: {latent_stats['latent_dim']}")
#         print(f"  PC1-3 coverage: {latent_stats['pc3_coverage']:.3f}")
#         print(f"  PC1-5 coverage: {latent_stats['pc5_coverage']:.3f}")

#     print("\n===========================================")


# if __name__ == "__main__":
#     main()
# '''
# python /home/zhaodanqi/clone/WoTE/ControllerExp/scripts/analysisStyleCoverage.py \
#    --style_npz /home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz
# '''



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Style Coverage Analysis for Controller Styles
======================================================

This version includes:

(A) Trajectory Behavior Diversity (pairwise L2)
(B) Clustering-based Coverage (KMeans + silhouette)
(C) Latent Space Coverage using ControllerEmbedding (PAC)

Usage:
python analysisStyleCoverage.py \
    --style_npz /home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
    --use_latent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style Coverage Analysis for Controller Styles (Complete)
========================================================

Computes three metrics for controller styles:

(A) Trajectory Behavior Diversity (pairwise distance)
(B) Clustering-based Coverage (KMeans + silhouette)
(C) Latent Space Coverage (PAC) using ControllerEmbedding

Handles automatic broadcasting of reference trajectories for PAC.

Author: WoTE / Controller-as-Latent Variable
"""
import argparse
import os
import sys
import numpy as np
import torch
from typing import Dict, Any
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ------------------------
# Load ControllerEmbedding
# ------------------------
sys.path.append("/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step2_Embedding/")
from ControllerEmbedding import ControllerEmbedding

# ------------------------
# Descriptor Definition (Metric A/B)
# ------------------------
def traj_descriptor(exec_traj: np.ndarray) -> np.ndarray:
    """
    exec_traj: [256, 8, 3] (single style)
    returns: 4-dim descriptor
    """
    xy = exec_traj[..., :2]
    yaw = exec_traj[..., 2]

    lateral_std = xy[..., 0].std()
    longitudinal_std = xy[..., 1].std()
    yaw_rate = np.abs(np.diff(yaw, axis=1)).mean()
    total_heading_change = np.abs(yaw[:, -1] - yaw[:, 0]).mean()

    return np.array([lateral_std, longitudinal_std, yaw_rate, total_heading_change], dtype=np.float64)

# ------------------------
# Metric A: Pairwise Behavior Diversity
# ------------------------
def compute_pairwise_coverage(descriptors: np.ndarray) -> Dict[str, float]:
    pairwise_dist = pdist(descriptors, metric="euclidean")
    return {
        "pairwise_mean": float(pairwise_dist.mean()),
        "pairwise_min": float(pairwise_dist.min()),
        "pairwise_max": float(pairwise_dist.max()),
    }

# ------------------------
# Metric B: Clustering Coverage
# ------------------------
def compute_clustering_coverage(descriptors: np.ndarray, k_list=(4,6,8,10), seed=0) -> Dict[str, Any]:
    results = {}
    for k in k_list:
        if k >= len(descriptors):
            continue
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(descriptors)
        sil = silhouette_score(descriptors, labels)
        results[f"k={k}"] = {"silhouette": float(sil), "inertia": float(kmeans.inertia_)}
    return results

# ------------------------
# Metric C: Latent PAC coverage
# ------------------------
def compute_latent_coverage(ref_trajs: np.ndarray, exec_trajs: np.ndarray, emb_dim=64, device="cpu") -> Dict[str, Any]:
    """
    ref_trajs: [256, 8, 3] or [N_ref, 8, 3]
    exec_trajs: [N_kind, 256, 8, 3]
    """
    N_kind, N_traj, N_step, D = exec_trajs.shape

    # Broadcast reference to [N_kind, 256, 8, 3]
    if ref_trajs.ndim == 3:
        ref_trajs_batch = np.repeat(ref_trajs[None, ...], N_kind, axis=0)
    else:
        raise ValueError(f"Unexpected ref_trajs shape: {ref_trajs.shape}")

    # Flatten 256 trajs per style into batch
    exec_flat = exec_trajs.reshape(N_kind*N_traj, N_step, D)
    ref_flat  = ref_trajs_batch.reshape(N_kind*N_traj, N_step, D)

    exec_tensor = torch.from_numpy(exec_flat).float().to(device)
    ref_tensor  = torch.from_numpy(ref_flat).float().to(device)

    # Initialize embedding
    encoder = ControllerEmbedding(emb_dim=emb_dim).to(device)
    encoder.eval()

    with torch.no_grad():
        latents_flat = encoder(ref_tensor, exec_tensor).cpu().numpy()  # [N_kind*256, emb_dim]

    # Average embedding per controller style
    latents = latents_flat.reshape(N_kind, N_traj, emb_dim).mean(axis=1)  # [N_kind, emb_dim]

    # PCA coverage
    pca = PCA()
    pca.fit(latents)
    explained = np.cumsum(pca.explained_variance_ratio_)

    # Covariance determinant
    cov_det = np.linalg.det(np.cov(latents.T) + 1e-6 * np.eye(latents.shape[1]))

    return {
        "latent_dim": latents.shape[1],
        "pca_explained_cumsum": explained.tolist(),
        "pc3_coverage": float(explained[min(2,len(explained)-1)]),
        "pc5_coverage": float(explained[min(4,len(explained)-1)]),
        "latent_cov_det": float(cov_det),
        "latent_std_mean": float(latents.std(axis=0).mean())
    }

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser("Controller Style Coverage Analysis")
    parser.add_argument("--style_npz", type=str, required=True, help="Path to controller_styles.npz")
    parser.add_argument("--ref_traj", type=str, default="/home/zhaodanqi/clone/WoTE/ControllerExp/refs/Anchors_Original_256_centered.npy",
                        help="Reference trajectory (for PAC embedding)")
    parser.add_argument("--k_list", type=int, nargs="+", default=[4,6,8,10])
    parser.add_argument("--use_latent", action="store_true", help="Compute latent PAC coverage using ControllerEmbedding")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for latent computation")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension for ControllerEmbedding")
    args = parser.parse_args()

    # ------------------------
    # Load exec_trajs
    # ------------------------
    assert os.path.exists(args.style_npz), f"File not found: {args.style_npz}"
    data = np.load(args.style_npz, allow_pickle=True)
    assert "exec_trajs" in data, "exec_trajs not found in npz"
    exec_trajs = data["exec_trajs"]  # [N_kind, 256, 8, 3]

    print(f"[INFO] Loaded {len(exec_trajs)} controller styles")

    # Metric A/B descriptors
    descriptors = np.stack([traj_descriptor(exec_trajs[i]) for i in range(len(exec_trajs))])
    print("[INFO] Descriptor shape:", descriptors.shape)

    # Metric A
    pairwise_stats = compute_pairwise_coverage(descriptors)

    # Metric B
    cluster_stats = compute_clustering_coverage(descriptors, k_list=args.k_list)

    # Metric C (optional)
    latent_stats = None
    if args.use_latent:
        assert os.path.exists(args.ref_traj), f"Ref traj not found: {args.ref_traj}"
        ref_trajs = np.load(args.ref_traj)  # [256,8,3]
        latent_stats = compute_latent_coverage(ref_trajs, exec_trajs, emb_dim=args.emb_dim, device=args.device)

    # ------------------------
    # Print summary
    # ------------------------
    print("\n========== Style Coverage Summary ==========")
    print("\n[Metric A] Pairwise Behavior Diversity")
    for k,v in pairwise_stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Metric B] Clustering-based Coverage")
    for k,stats in cluster_stats.items():
        print(f"  {k}: silhouette={stats['silhouette']:.3f}, inertia={stats['inertia']:.2f}")

    if latent_stats is not None:
        print("\n[Metric C] Latent Space Coverage (PAC)")
        print(f"  latent_dim: {latent_stats['latent_dim']}")
        print(f"  PC1-3 coverage: {latent_stats['pc3_coverage']:.3f}")
        print(f"  PC1-5 coverage: {latent_stats['pc5_coverage']:.3f}")
        print(f"  latent_cov_det: {latent_stats['latent_cov_det']:.4e}")
        print(f"  latent_std_mean: {latent_stats['latent_std_mean']:.4f}")

    print("\n===========================================")

if __name__ == "__main__":
    main()

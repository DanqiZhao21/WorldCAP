import numpy as np

from tool.visualize.wote_pdm_analysis import (
    compute_spearman_rho,
    plot_sorted_distribution,
    save_analysis_artifacts,
    rank_trajectories_left_to_right,
)


def test_compute_spearman_rho_perfect_inverse():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    rho = compute_spearman_rho(x, y)
    assert rho == -1.0


def test_rank_trajectories_left_to_right_orders_by_yaw_then_lateral_offset():
    trajectories = np.array(
        [
            [[0.0, 0.0, -0.6], [4.0, -2.0, -0.6]],  # right-turn
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],      # straight
            [[0.0, 0.0, 0.6], [4.0, 2.0, 0.6]],      # left-turn
        ],
        dtype=np.float32,
    )
    order = rank_trajectories_left_to_right(trajectories)
    assert order.tolist() == [2, 1, 0]


def test_save_analysis_artifacts_writes_csv_and_npz(tmp_path):
    trajectories = np.array(
        [
            [[0.0, 0.0, -0.5], [1.0, -1.0, -0.5]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    data = {
        "rows": [
            {"index": 0, "wote_score": 0.1, "pdm_score": 0.2},
            {"index": 1, "wote_score": 0.3, "pdm_score": 0.4},
        ],
        "trajectories": trajectories,
        "wote_scores": np.array([0.1, 0.3], dtype=np.float32),
        "pdm_scores": np.array([0.2, 0.4], dtype=np.float32),
        "rank_order": np.array([0, 1], dtype=np.int64),
        "spearman_rho": 1.0,
    }
    csv_path, npz_path = save_analysis_artifacts(tmp_path, data)
    assert csv_path.exists()
    assert npz_path.exists()


def test_plot_sorted_distribution_creates_file(tmp_path):
    trajectories = np.array(
        [
            [[0.0, 0.0, -0.5], [1.0, -1.0, -0.5]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.5], [1.0, 1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    out = plot_sorted_distribution(
        tmp_path / "plot.png",
        trajectories=trajectories,
        wote_scores=np.array([0.2, 0.5, 0.8], dtype=np.float32),
        pdm_scores=np.array([0.9, 0.6, 0.1], dtype=np.float32),
    )
    assert out.exists()

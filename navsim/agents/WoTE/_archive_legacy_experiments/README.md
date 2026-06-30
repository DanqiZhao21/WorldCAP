Legacy WoTE Experiment Snapshots
================================

This directory holds old one-off experiment snapshots that used to live next to
the active WoTE implementation. They are not imported by the main NAVSIM WoTE
agent.

The current paper/mainline path is:

- `../WoTE_model.py`
- `../WoTE_loss.py`
- `../WoTE_targets.py`
- `../configs/default.py`

These archived files are kept only for provenance. In particular, they may
contain older controller-to-offset, controller-to-reward, response-predictor,
HERM execution-predictor, or candidate-relabeling experiments. Those paths are
not part of the cleaned controller-conditioned BEV world model setting.

from navsim.agents.WoTE.configs.default import WoTEConfig


def test_wote_config_declares_yaml_loss_switches():
    fields = WoTEConfig.__dataclass_fields__

    assert "use_agent_loss" in fields
    assert "use_map_loss" in fields


def test_wote_config_separates_style_observations_from_candidate_executions():
    fields = WoTEConfig.__dataclass_fields__

    assert "controller_style_ref_bank_path" in fields
    assert "controller_style_exec_bank_path" in fields
    assert "controller_candidate_exec_bank_path" in fields
    assert "cluster_file_path" in fields

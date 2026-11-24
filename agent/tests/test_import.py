from importlib.metadata import version


def test_agent_installation():
    import gymnasium
    import mun_agent
    import torch
    from mun_lander import LanderEnv

    assert mun_agent.__package__ == "mun_agent"
    assert version("mun-agent")
    assert issubclass(LanderEnv, gymnasium.Env)
    assert torch.zeros(1, device="cpu").item() == 0

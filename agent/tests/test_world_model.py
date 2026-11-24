import pytest
import torch
from torch.nn import functional as F

from mun_agent.world_model import WorldModel, decode_two_hot, kl_losses, symexp, symlog, two_hot


@pytest.fixture
def model():
    torch.manual_seed(73)
    return WorldModel(deter=12, stoch=3, classes=4, hidden=16, bins=51)


def test_symlog_two_hot_roundtrip_and_endpoints():
    centers = torch.linspace(-20, 20, 255)
    x = torch.tensor([-1.0e8, -100.0, -0.01, 0.0, 0.01, 100.0, 1.0e8])
    torch.testing.assert_close(symexp(symlog(x)), x)
    encoded = two_hot(x, centers)
    assert (encoded >= 0).all()
    assert ((encoded > 0).sum(-1) <= 2).all()
    torch.testing.assert_close(encoded.sum(-1), torch.ones_like(x))
    torch.testing.assert_close(decode_two_hot(encoded, centers), x, rtol=2e-5, atol=2e-6)
    clipped = two_hot(torch.tensor([-1e14, 1e14]), centers)
    assert clipped[0, 0] == 1 and clipped[1, -1] == 1


def test_kl_free_nats_and_gradient_routing():
    qlogit = torch.tensor([[[4., -4.], [4., -4.]]], requires_grad=True)
    plogit = (-qlogit.detach()).requires_grad_()
    q, p = qlogit.softmax(-1), plogit.softmax(-1)
    dyn, rep = kl_losses(q, p)
    assert dyn.shape == (1,) and dyn.item() > 1
    torch.testing.assert_close(dyn, rep)
    dq, dp = torch.autograd.grad(dyn.sum(), (qlogit, plogit), allow_unused=True)
    assert dq is None and dp.abs().sum() > 0
    dq, dp = torch.autograd.grad(rep.sum(), (qlogit, plogit), allow_unused=True)
    assert dp is None and dq.abs().sum() > 0

    q = torch.tensor([[[0.51, 0.49]]], requires_grad=True)
    p = torch.tensor([[[0.50, 0.50]]], requires_grad=True)
    dyn, rep = kl_losses(q, p)
    assert dyn.item() == rep.item() == 1.0
    (dyn + rep).sum().backward()
    assert not q.grad.any() and not p.grad.any()


def test_rssm_shapes_and_mode(model):
    features = torch.randn(5, 16)
    actions = torch.randn(5, 4).tanh()
    state = model.observe_initial(features)
    assert state[0].shape == (5, 12)
    assert state[1].shape == (5, 3, 4)
    successor = model.observe_step(state, actions, -features)
    imagined = model.imagine_step(state, actions)
    torch.testing.assert_close(imagined[0], successor[0])
    for h, z in (state, successor, imagined, model.imagine_step(state, actions, sample=True)):
        assert ((z == 1) | (z == 0)).all()
        assert (z.sum(-1) == 1).all()
        decoded, reward, cont = model.predict((h, z))
        assert decoded.shape == (5, 16) and reward.shape == cont.shape == (5,)
        assert ((0 < cont) & (cont < 1)).all()
    torch.testing.assert_close(imagined[1], model.imagine_step(state, actions)[1])


def test_reconstruction_reward_and_continue_align_with_successor(model):
    features = torch.randn(2, 4, 16)
    actions = torch.randn(2, 3, 4)
    rewards = torch.tensor([[0.2, 0.6, -100.], [-0.3, -0.7, 100.]])
    done = torch.tensor([[False, False, True], [False, False, True]])
    # Make reward logits state-dependent even at initialization.
    torch.nn.init.normal_(model.reward_head[-1].weight)
    losses = model.loss(features, actions, rewards, done, torch.ones_like(done), sample=False)
    state = model.observe_initial(features[:, 0])
    expected = {name: [] for name in ("reconstruction", "reward", "continue")}
    for t in range(3):
        state = model.observe_step(state, actions[:, t], features[:, t + 1])
        est, _, cont = model.predict(state)
        expected["reconstruction"].append(F.mse_loss(symlog(est), symlog(features[:, t + 1])))
        latent = torch.cat((state[0], state[1].flatten(-2)), -1)
        reward_logprob = model.reward_head(latent).log_softmax(-1)
        expected["reward"].append(-(two_hot(rewards[:, t], model.centers) * reward_logprob).sum(-1).mean())
        expected["continue"].append(F.binary_cross_entropy(cont, (~done[:, t]).float()))
    for name, steps in expected.items():
        torch.testing.assert_close(losses[name], torch.stack(steps).mean())


def test_left_and_right_padding_leave_losses_unchanged(model):
    x, a = torch.randn(1, 4, 16), torch.randn(1, 3, 4)
    r, terminal = torch.randn(1, 3), torch.tensor([[False, False, True]])
    mask = torch.ones_like(terminal)

    def pad(values, left, right):
        before = values.new_ones((1, left, *values.shape[2:]))
        after = values.new_ones((1, right, *values.shape[2:]))
        if values.dtype != torch.bool:
            before.fill_(float("nan"))
            after.fill_(float("nan"))
        return torch.cat((before, values, after), 1)

    torch.manual_seed(44)
    expected = model.loss(x, a, r, terminal, mask)
    padded = [pad(values, 2, 3) for values in (x, a, r, terminal)]
    padded_mask = F.pad(mask, (2, 3), value=False)
    torch.manual_seed(44)
    got = model.loss(*padded, padded_mask)
    for key in expected:
        torch.testing.assert_close(expected[key], got[key])
    got["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_can_overfit_tiny_transition_batch(model):
    x = torch.ones(2, 4, 16)
    a = torch.zeros(2, 3, 4)
    r = torch.full((2, 3), 0.5)
    terminal = torch.zeros((2, 3), dtype=torch.bool)
    mask = torch.ones_like(terminal)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.no_grad():
        start = model.loss(x, a, r, terminal, mask, sample=False)["loss"].item()
    for _ in range(35):
        optimizer.zero_grad()
        model.loss(x, a, r, terminal, mask)["loss"].backward()
        optimizer.step()
    finish = model.loss(x, a, r, terminal, mask, sample=False)["loss"].item()
    assert finish < 0.65 * start

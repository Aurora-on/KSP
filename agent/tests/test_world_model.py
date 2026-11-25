import pytest
import torch
from torch.nn import functional as F

from mun_agent.world_model import WorldModel, decode_two_hot, kl_losses, symexp, symlog, two_hot


@pytest.fixture
def model():
    torch.manual_seed(73)
    return WorldModel(deter=12, stoch=3, classes=4, hidden=16, bins=51)


def test_symexp_two_hot_roundtrip_and_endpoints(model):
    centers = model.reward_bins
    x = torch.tensor([-1.0e8, -100.0, -0.01, 0.0, 0.01, 100.0, 1.0e8])
    torch.testing.assert_close(symexp(symlog(x)), x)
    encoded = two_hot(x, centers)
    assert (encoded >= 0).all()
    assert ((encoded > 0).sum(-1) <= 2).all()
    torch.testing.assert_close(encoded.sum(-1), torch.ones_like(x))
    torch.testing.assert_close(decode_two_hot(encoded, centers), x, rtol=2e-5, atol=2e-6)
    clipped = two_hot(torch.tensor([-1e14, 1e14]), centers)
    assert clipped[0, 0] == 1 and clipped[1, -1] == 1


def test_two_hot_interpolates_in_original_reward_units(model):
    bins = model.reward_bins
    index = len(bins) // 2 + 5
    target = 0.25 * bins[index] + 0.75 * bins[index + 1]
    weights = two_hot(target, bins)
    torch.testing.assert_close(weights[index:index + 2], torch.tensor([0.25, 0.75]))
    assert torch.count_nonzero(weights) == 2


@pytest.mark.parametrize("bins", [254, 255])
def test_multimodal_reward_mean_from_posterior_prediction(bins):
    model = WorldModel(12, 3, 4, 16, bins)
    centers = model.reward_bins
    torch.testing.assert_close(centers, -centers.flip((0,)), rtol=0, atol=0)
    target = torch.tensor([-120., -1., 0., 100.])
    mixture = torch.tensor([0.18, 0.27, 0.30, 0.25])
    # A batch with these frequencies has a cross-entropy optimum equal to the
    # weighted sum of its encoded targets. Test that readout via the real head.
    probabilities = (mixture[:, None] * two_hot(target, centers)).sum(0)
    with torch.no_grad():
        model.reward_head[-1].bias.copy_(probabilities.log())
    state = model.observe_initial(torch.ones(2, 16))
    expected = (mixture * target).sum().expand(2)
    torch.testing.assert_close(model.predict(state)[1], expected, atol=2e-5, rtol=0)

    # Symmetric tails can be ~1e8 times larger than an interior reward.
    # The interior contribution must survive cancellation on both grids.
    near_zero = torch.zeros(bins)
    near_zero[0] = near_zero[-1] = 0.49
    near_zero[bins // 2 + 1] = 0.02
    torch.testing.assert_close(decode_two_hot(near_zero, centers),
                               0.02 * centers[bins // 2 + 1], rtol=0, atol=0)
    flat = torch.ones(bins) / bins
    assert decode_two_hot(flat, centers).item() == 0


def test_format_one_reward_head_reproduces_original_readout():
    old = WorldModel(12, 3, 4, 16, 51, legacy_reward=True)
    with torch.no_grad():
        old.reward_head[-1].bias.copy_(torch.linspace(-1, 0, 51))
    state = old.observe_initial(torch.randn(2, 16))
    _, logits, _ = old._heads(state)
    expected = symexp((logits.softmax(-1) * torch.linspace(-20, 20, 51)).sum(-1))
    torch.testing.assert_close(old.predict(state)[1], expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="inference only"):
        old.loss(None, None, None, None, None)
    with pytest.raises(RuntimeError, match="reward_bins"):
        WorldModel(12, 3, 4, 16, 51).load_state_dict(old.state_dict())


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
        expected["reconstruction"].append((symlog(est) - symlog(features[:, t + 1])).square().sum(-1).mean())
        latent = torch.cat((state[0], state[1].flatten(-2)), -1)
        reward_logprob = model.reward_head(latent).log_softmax(-1)
        expected["reward"].append(-(two_hot(rewards[:, t], model.reward_bins) * reward_logprob).sum(-1).mean())
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


def test_ground_history_only_enters_posterior_and_event_weight_only_changes_head_losses():
    torch.manual_seed(17)
    kwargs = dict(deter=12, stoch=3, classes=4, hidden=16, bins=51, ground_history=True)
    model = WorldModel(**kwargs, event_weight=12, dynamics_scale=0.2,
                       representation_scale=0.01, free_nats=0.0)
    x = torch.randn(3, 3, 17)
    a = torch.randn(3, 2, 4)
    done = torch.tensor([[False, False], [False, True], [True, False]])
    mask = torch.ones_like(done)
    rewards = torch.where(done, 100., 0.)
    torch.nn.init.normal_(model.reward_head[-1].weight)
    losses = model.loss(x, a, rewards, done, mask, sample=False)
    state = model.observe_initial(x[:, 0])
    ces, bces = [], []
    for t in range(2):
        state = model.observe_step(state, a[:, t], x[:, t+1])
        decoded, _, probs = model.predict(state)
        assert decoded.shape[-1] == 16
        raw_logits = model.reward_head(torch.cat((state[0], state[1].flatten(-2)), -1))
        ces.append(-(two_hot(rewards[:, t], model.reward_bins) * raw_logits.log_softmax(-1)).sum(-1))
        bces.append(F.binary_cross_entropy(probs, (~done[:, t]).float(), reduction="none"))
    weights = torch.where(done, 12., 1.)
    torch.testing.assert_close(losses["reward"], (weights * torch.stack(ces, 1)).sum() / weights.sum())
    torch.testing.assert_close(losses["continue"], (weights * torch.stack(bces, 1)).sum() / weights.sum())
    expected = (losses["reconstruction"] + losses["reward"] + losses["continue"]
                + 0.2 * losses["dynamics"] + 0.01 * losses["representation"])
    torch.testing.assert_close(losses["loss"], expected)
    assert model.imagine_step(state, a[:, 0])[0].shape == (3, 12)


def test_fit_reduces_reconstruction_and_position_error_vs_training_mean(model):
    # Four known physical x positions in metres; the encoder sees scaled values.
    position_m = torch.tensor([-60., -20., 20., 60.])
    x = torch.zeros(4, 4, 16)
    x[..., 0] = position_m[:, None] / 20.
    a = torch.zeros(4, 3, 4)
    r = torch.zeros(4, 3)
    done = torch.zeros_like(r, dtype=torch.bool)
    mask = torch.ones_like(done)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    with torch.no_grad():
        start = model.loss(x, a, r, done, mask, sample=False)["reconstruction"].item()
    for _ in range(95):
        optimizer.zero_grad()
        model.loss(x, a, r, done, mask)["loss"].backward()
        optimizer.step()
    finish = model.loss(x, a, r, done, mask, sample=False)["reconstruction"].item()
    state = model.observe_initial(x[:, 0])
    errors = []
    for t in range(3):
        state = model.observe_step(state, a[:, t], x[:, t+1])
        decoded, _, _ = model.predict(state)
        errors.append((20 * decoded[:, 0] - position_m).abs())
    mean_baseline = (position_m.mean() - position_m).abs().mean()
    assert finish < 0.5 * start
    assert torch.stack(errors).mean() < 0.6 * mean_baseline

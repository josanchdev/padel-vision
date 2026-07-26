import numpy as np
import torch
from padel_ml.data import load_shot_data
from padel_ml.graph import NUM_JOINTS, normalized_adjacency
from padel_ml.stgcn import STGCN


def test_adjacency_is_symmetric_and_normalized() -> None:
    adj = normalized_adjacency()
    assert adj.shape == (NUM_JOINTS, NUM_JOINTS)
    np.testing.assert_allclose(adj, adj.T, atol=1e-6)
    assert (np.diag(adj) > 0).all()  # self-loops present


def test_stgcn_forward_shape() -> None:
    model = STGCN(num_classes=6)
    out = model(torch.randn(4, 3, 32, 17))
    assert out.shape == (4, 6)


def test_stgcn_is_trainable() -> None:
    model = STGCN(num_classes=6)
    x = torch.randn(2, 3, 32, 17)
    y = torch.tensor([0, 3])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_load_shot_data_splits_by_match(tmp_path) -> None:
    n = 20
    npz = tmp_path / "clips.npz"
    np.savez_compressed(
        npz,
        keypoints=np.random.rand(n, 32, 17, 3).astype(np.float32),
        labels=np.array([i % 6 for i in range(n)], dtype=np.int64),
        matches=np.array([0] * 10 + [1] * 10, dtype=np.int64),
        classes=np.array(["Forehand", "Backhand", "Smash", "Serve", "Other", "NoShot"]),
    )
    data = load_shot_data(npz, val_match=0)
    assert len(data.val) == 10  # match 0 held out
    assert len(data.train) == 10
    x, _ = data.train[0]
    assert x.shape == (3, 32, 17)
    assert data.class_weights.shape == (6,)

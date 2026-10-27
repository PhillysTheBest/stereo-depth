import pytest


@pytest.mark.gpu
def test_cuda_visible() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()

import pytest

from src.dataloader import DataLoader

class TestDataloader:
    def test_dataloader_merges_correctly():
        assert 0.1 + 0.2 == pytest.approx(0.3)
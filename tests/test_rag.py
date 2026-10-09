import numpy as np
import pytest

from petrec.rag import NumpyVectorStore, chunk_text


def test_chunking_and_numpy_cosine_retrieval():
    chunks = chunk_text("word " * 220, chunk_size=400, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)
    store = NumpyVectorStore()
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    store.add(["aligned", "orthogonal"], ["a", "b"], vectors)
    found = store.search(vectors[0], top_m=1)
    assert found[0].id == "aligned"
    assert found[0].score == pytest.approx(1.0)

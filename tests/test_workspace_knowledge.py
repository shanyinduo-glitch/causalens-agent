import json

from causalens.ui import workspace_knowledge as wk


def test_workspace_knowledge_is_scoped_and_persisted(tmp_path, monkeypatch):
    captured = {}

    def fake_build_vector_store(**kwargs):
        captured.update(kwargs)
        return {"stored_count": len(kwargs["chunks"]), "collection_name": kwargs["collection_name"]}

    monkeypatch.setattr(wk, "build_vector_store", fake_build_vector_store)
    dataset_sha = "a" * 64
    manifest = wk.build_workspace_knowledge(
        tmp_path,
        dataset_sha,
        "研究对象是黄金价格。TIPS实际收益率变化是处理变量，黄金收益率是结果变量。",
        dataset_context="# 字段\n- t_diff（float64）\n- y_ret（float64）",
    )

    assert manifest["dataset_sha256"] == dataset_sha
    assert manifest["collection_name"].startswith("workspace_")
    assert manifest["chunk_count"] == 2
    assert manifest["document_count"] == 2
    assert captured["collection_name"] == manifest["collection_name"]
    assert wk.load_workspace_manifest(tmp_path, dataset_sha) == manifest
    records = wk.load_workspace_records(tmp_path, dataset_sha)
    assert records and "黄金价格" in records[0]["text"]
    assert any("t_diff" in record["text"] for record in records)


def test_workspace_knowledge_retrieval_returns_cited_local_source(tmp_path, monkeypatch):
    dataset_sha = "b" * 64
    target = tmp_path / "storage" / "workspace_knowledge" / dataset_sha
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(
        json.dumps({
            "dataset_sha256": dataset_sha,
            "collection_name": wk.collection_name(dataset_sha),
            "embedding_model": "test-model",
            "chunk_count": 1,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        wk,
        "retrieve_documents",
        lambda **kwargs: [{
            "chunk_id": "workspace_bbbb_0000",
            "section": "研究对象",
            "content": "TIPS实际收益率变化与黄金收益率是本案例的研究变量。",
            "distance": 0.12,
        }],
    )

    results = wk.retrieve_workspace_knowledge(tmp_path, dataset_sha, "黄金收益率")
    assert results[0]["source"] == "用户建立的本地知识库"
    assert results[0]["id"].startswith("K-")
    assert "黄金收益率" in results[0]["text"]


def test_workspace_knowledge_falls_back_when_embedding_model_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wk,
        "build_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )
    dataset_sha = "f" * 64
    manifest = wk.build_workspace_knowledge(
        tmp_path,
        dataset_sha,
        "研究对象是黄金价格，TIPS收益率变化是处理变量，黄金收益率是结果变量。",
    )
    assert manifest["storage_mode"] == "lexical_fallback"
    results = wk.retrieve_workspace_knowledge(tmp_path, dataset_sha, "黄金收益率")
    assert "词项检索" in results[0]["source"]


def test_workspace_knowledge_lexical_fallback_supports_chinese_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wk,
        "build_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    dataset_sha = "d" * 64
    wk.build_workspace_knowledge(
        tmp_path,
        dataset_sha,
        "研究对象是黄金价格，TIPS实际收益率变化是处理变量，黄金收益率是结果变量。",
    )
    results = wk.retrieve_workspace_knowledge(tmp_path, dataset_sha, "TIPS收益率和黄金价格")
    assert results
    assert "用户研究背景" in results[0]["source"]


def test_delete_workspace_knowledge_removes_dataset_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wk,
        "build_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    dataset_sha = "e" * 64
    wk.build_workspace_knowledge(tmp_path, dataset_sha, "这是足够长的研究背景，用于测试删除当前数据版本知识库。")
    assert wk.delete_workspace_knowledge(tmp_path, dataset_sha) is True
    assert wk.load_workspace_manifest(tmp_path, dataset_sha) is None
    assert wk.load_workspace_records(tmp_path, dataset_sha) == []


def test_workspace_knowledge_rejects_short_background(tmp_path):
    try:
        wk.build_workspace_knowledge(tmp_path, "c" * 64, "太短")
    except ValueError as exc:
        assert "20 个字符" in str(exc)
    else:
        raise AssertionError("short background should be rejected")


def test_workspace_knowledge_can_start_with_dataset_schema_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wk,
        "build_vector_store",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    manifest = wk.build_workspace_knowledge(
        tmp_path,
        "1" * 64,
        "",
        dataset_context="# 字段\n- date（str）\n- y_ret（float64）",
    )
    assert manifest["document_count"] == 1
    assert wk.load_workspace_records(tmp_path, "1" * 64)[0]["source"] == "当前数据集结构与概况"

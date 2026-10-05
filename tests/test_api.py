from app.context import NOT_FOUND_ANSWER
from app.retrieval.retriever import Retriever
from tests.conftest import STUDY_TEXT


def upload(client, auth, name="study.txt", content=STUDY_TEXT):
    return client.post("/documents", headers=auth, files={"file": (name, content.encode())})


def test_health_is_public(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["llm_provider"] == "fake"


def test_protected_routes_need_the_api_key(client):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json={"question": "anything?"}).status_code == 401


def test_upload_then_list(client, auth):
    created = upload(client, auth)

    assert created.status_code == 201
    body = created.json()
    assert body["pages"] == 2 and body["chunks"] >= 2 and body["created"]
    assert [d["id"] for d in client.get("/documents", headers=auth).json()] == [body["id"]]


def test_same_file_is_not_ingested_twice(client, auth):
    first = upload(client, auth).json()
    second = upload(client, auth, name="renamed.txt")

    assert second.status_code == 200
    assert second.json()["id"] == first["id"]
    assert second.json()["created"] is False


def test_unsupported_and_empty_uploads_are_rejected(client, auth):
    assert upload(client, auth, name="image.png").status_code == 415
    assert upload(client, auth, name="blank.txt", content="   ").status_code == 422


def test_ask_returns_answer_with_resolved_citations_and_a_trace(client, auth, llm):
    upload(client, auth)
    llm.replies = ["No deaths occurred during the study [1]."]

    response = client.post(
        "/ask", headers=auth, json={"question": "Were there deaths during the study?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"][0]["filename"] == "study.txt"
    assert body["citations"][0]["page"] in (1, 2)

    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "No deaths occurred" in prompt  # the retrieved passage reached the model

    trace = client.get(f"/traces/{body['trace_id']}", headers=auth).json()
    assert [s["step"] for s in trace["steps"]] == [
        "rewrite_query",
        "retrieve",
        "generate",
        "check_citations",
    ]
    assert trace["prompt_versions"][0].startswith("qa_system@")
    assert response.headers["X-Request-ID"]


def test_unrelated_question_is_refused_without_calling_the_model(client, auth, llm):
    upload(client, auth)

    response = client.post("/ask", headers=auth, json={"question": "Best sourdough recipe?"})

    assert response.json()["answer"] == NOT_FOUND_ANSWER
    assert response.json()["citations"] == []
    assert llm.calls == []


def test_answer_without_citations_is_flagged(client, auth, llm):
    upload(client, auth)
    llm.replies = ["There were 240 participants."]

    response = client.post(
        "/ask", headers=auth, json={"question": "How many participants were randomized?"}
    )

    assert response.json()["grounded"] is False


def test_follow_up_question_is_rewritten_before_search(client, auth, llm):
    upload(client, auth)
    llm.replies = ["How many participants reported headache in study ABC-1?", "Twelve [1]."]

    response = client.post(
        "/ask",
        headers=auth,
        json={
            "question": "And how many had that?",
            "history": [
                {"role": "user", "content": "Which adverse events were reported?"},
                {"role": "assistant", "content": "Headache [1]."},
            ],
        },
    )

    trace = client.get(f"/traces/{response.json()['trace_id']}", headers=auth).json()
    assert trace["search_query"] == "How many participants reported headache in study ABC-1?"
    assert len(llm.calls) == 2


def test_question_can_be_scoped_to_one_document(client, auth):
    doc = upload(client, auth).json()

    ok = client.post(
        "/ask", headers=auth, json={"question": "deaths?", "document_ids": [doc["id"]]}
    )
    missing = client.post("/ask", headers=auth, json={"question": "deaths?", "document_ids": ["x"]})

    assert ok.status_code == 200
    assert missing.status_code == 404


def test_delete_removes_document_and_its_chunks(client, auth, container):
    doc = upload(client, auth).json()

    assert client.delete(f"/documents/{doc['id']}", headers=auth).status_code == 204
    assert client.get("/documents", headers=auth).json() == []
    assert container.store.count() == 0
    assert client.delete(f"/documents/{doc['id']}", headers=auth).status_code == 404


def test_reranker_decides_the_order_but_scores_stay_cosine(container, client, auth):
    upload(client, auth)

    class PreferSafety:
        def scores(self, query, passages):
            return [1.0 if "Headache" in p else 0.0 for p in passages]

    retriever = Retriever(container.store, mode="rerank", reranker=PreferSafety())
    hits = retriever.search("participants randomized primary endpoint", top_k=2)

    assert "Headache" in hits[0].text
    assert all(0.0 <= h.score <= 1.0 for h in hits)

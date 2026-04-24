from app.llm.response_utils import coerce_llm_text
from app.rag.chains import FilingQAChain


def test_coerce_llm_text_handles_provider_blocks():
    response = type("Response", (), {"content": [{"type": "text", "text": "Hello world"}]})()
    assert coerce_llm_text(response) == "Hello world"


def test_chain_coerce_response_text_handles_provider_blocks():
    response = type("Response", (), {"content": [{"type": "text", "text": "Answer text"}]})()
    assert FilingQAChain._coerce_response_text(response) == "Answer text"

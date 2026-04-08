import os
import json
import tempfile
from mempalace.normalize import normalize


def test_plain_text():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("Hello world\nSecond line\n")
    f.close()
    result = normalize(f.name)
    assert "Hello world" in result
    os.unlink(f.name)


def test_claude_json():
    data = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    assert "Hi" in result
    os.unlink(f.name)


def test_empty():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.close()
    result = normalize(f.name)
    assert result.strip() == ""
    os.unlink(f.name)


def test_claude_ai_sender_field():
    """Claude.ai exports use 'sender' (human/assistant), not 'role' (user/assistant)."""
    data = [
        {
            "uuid": "conv-1",
            "name": "Test Chat",
            "chat_messages": [
                {"sender": "human", "text": "What is context engineering?"},
                {"sender": "assistant", "text": "Context engineering is designing LLM prompts."},
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert ">" in result, "Should produce transcript with > markers"
    assert "context engineering" in result.lower()
    assert "Test Chat" in result, "Conversation name should appear as header"


def test_claude_ai_text_field_preferred():
    """Claude.ai 'text' field should be used even when 'content' is a block list."""
    data = [
        {
            "uuid": "conv-1",
            "name": "",
            "chat_messages": [
                {
                    "sender": "human",
                    "text": "Plain text question",
                    "content": [{"type": "text", "text": "Plain text question"}],
                },
                {
                    "sender": "assistant",
                    "text": "Plain text answer",
                    "content": [{"type": "text", "text": "Plain text answer"}],
                },
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Plain text question" in result
    assert "Plain text answer" in result


def test_claude_ai_multi_conversation_boundaries():
    """Multiple conversations should be separated, not merged into one blob."""
    data = [
        {
            "uuid": "conv-1",
            "name": "Swift Concurrency",
            "chat_messages": [
                {"sender": "human", "text": "How do actors work?"},
                {"sender": "assistant", "text": "Actors serialize access to mutable state."},
            ],
        },
        {
            "uuid": "conv-2",
            "name": "Kubernetes Setup",
            "chat_messages": [
                {"sender": "human", "text": "How to set up a cluster?"},
                {"sender": "assistant", "text": "Use kubeadm init to bootstrap."},
            ],
        },
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "Swift Concurrency" in result
    assert "Kubernetes Setup" in result
    assert "actors" in result.lower()
    assert "kubeadm" in result.lower()


def test_claude_ai_flat_sender_format():
    """Flat message list with 'sender' instead of 'role'."""
    data = [
        {"sender": "human", "text": "Hello"},
        {"sender": "assistant", "text": "Hi there!"},
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Hello" in result
    assert "Hi there!" in result


def test_claude_ai_role_field_still_works():
    """Existing 'role' field format should continue working."""
    data = [
        {
            "uuid": "conv-1",
            "name": "Test",
            "chat_messages": [
                {"role": "user", "content": "Question here"},
                {"role": "assistant", "content": "Answer here"},
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Question here" in result
    assert "Answer here" in result

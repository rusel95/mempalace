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


# ── Edge cases for Claude.ai normalizer robustness ──


def test_claude_ai_empty_chat_messages():
    """Conversations with no chat_messages should be skipped, not crash."""
    data = [
        {"uuid": "conv-1", "name": "Empty Conv", "chat_messages": []},
        {
            "uuid": "conv-2",
            "name": "Real Conv",
            "chat_messages": [
                {"sender": "human", "text": "Only real chat"},
                {"sender": "assistant", "text": "Only real response"},
            ],
        },
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Only real chat" in result
    assert "Empty Conv" not in result


def test_claude_ai_single_message_conversation():
    """A conversation with only one message (no reply) should be skipped."""
    data = [
        {
            "uuid": "conv-1",
            "name": "Orphan",
            "chat_messages": [
                {"sender": "human", "text": "Unanswered question"},
            ],
        },
        {
            "uuid": "conv-2",
            "name": "Complete",
            "chat_messages": [
                {"sender": "human", "text": "Answered question"},
                {"sender": "assistant", "text": "Here is the answer"},
            ],
        },
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    # The complete conversation should be present
    assert "Answered question" in result
    assert "Here is the answer" in result


def test_claude_ai_content_block_list_fallback():
    """When text is empty, fall back to content block list extraction."""
    data = [
        {
            "uuid": "conv-1",
            "name": "",
            "chat_messages": [
                {
                    "sender": "human",
                    "text": "",
                    "content": [{"type": "text", "text": "Extracted from blocks"}],
                },
                {
                    "sender": "assistant",
                    "text": "",
                    "content": [{"type": "text", "text": "Block response"}],
                },
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Extracted from blocks" in result
    assert "Block response" in result


def test_claude_ai_mixed_sender_and_role():
    """Export mixing sender and role fields (defensive — shouldn't happen but might)."""
    data = [
        {
            "uuid": "conv-1",
            "name": "",
            "chat_messages": [
                {"sender": "human", "text": "From sender field"},
                {"role": "assistant", "content": "From role field"},
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> From sender field" in result
    assert "From role field" in result


def test_claude_ai_unnamed_conversation_no_header():
    """Conversation with empty name should not produce a --- header."""
    data = [
        {
            "uuid": "conv-1",
            "name": "",
            "chat_messages": [
                {"sender": "human", "text": "No header expected"},
                {"sender": "assistant", "text": "Just content"},
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "---" not in result, "Empty name should not produce a separator header"
    assert "> No header expected" in result


def test_claude_ai_long_multi_turn_conversation():
    """Multi-turn conversation preserves all exchanges in correct order."""
    messages = []
    for i in range(10):
        messages.append({"sender": "human", "text": f"Question {i}"})
        messages.append({"sender": "assistant", "text": f"Answer {i}"})
    data = [{"uuid": "conv-1", "name": "Long Chat", "chat_messages": messages}]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    for i in range(10):
        assert f"Question {i}" in result
        assert f"Answer {i}" in result
    # Verify order: Question 0 before Question 9
    assert result.index("Question 0") < result.index("Question 9")


def test_claude_ai_whitespace_only_messages_skipped():
    """Messages with only whitespace should be skipped."""
    data = [
        {
            "uuid": "conv-1",
            "name": "",
            "chat_messages": [
                {"sender": "human", "text": "   "},
                {"sender": "assistant", "text": "  \n  "},
                {"sender": "human", "text": "Real question"},
                {"sender": "assistant", "text": "Real answer"},
            ],
        }
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Real question" in result
    assert "Real answer" in result


def test_chatgpt_conversations_json():
    """ChatGPT conversations.json with mapping tree still works after changes."""
    data = {
        "mapping": {
            "root": {
                "parent": None,
                "message": None,
                "children": ["msg1"],
            },
            "msg1": {
                "parent": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["ChatGPT user message"]},
                },
                "children": ["msg2"],
            },
            "msg2": {
                "parent": "msg1",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["ChatGPT assistant reply"]},
                },
                "children": [],
            },
        }
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> ChatGPT user message" in result
    assert "ChatGPT assistant reply" in result


def test_claude_code_jsonl_still_works():
    """Claude Code JSONL format should still work after changes."""
    lines = [
        json.dumps({"type": "human", "message": {"content": "Code question"}}),
        json.dumps({"type": "assistant", "message": {"content": "Code answer"}}),
    ]
    content = "\n".join(lines)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write(content)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "> Code question" in result
    assert "Code answer" in result


def test_slack_json_still_works():
    """Slack JSON export should still work after changes."""
    data = [
        {"type": "message", "user": "U123", "text": "Slack message 1"},
        {"type": "message", "user": "U456", "text": "Slack reply 1"},
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "Slack message 1" in result
    assert "Slack reply 1" in result

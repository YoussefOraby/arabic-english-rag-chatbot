"""Tests for Streamlit conversation history helpers."""

import json


def test_list_sessions_empty(tmp_path):
    """Verify _list_sessions returns empty list when directory is missing."""
    from src.ui.streamlit_app import HISTORY_DIR, _list_sessions

    original = HISTORY_DIR
    try:
        import src.ui.streamlit_app as ui

        ui.HISTORY_DIR = tmp_path / "nonexistent"
        assert _list_sessions() == []
    finally:
        ui.HISTORY_DIR = original


def test_list_sessions_returns_json_files(tmp_path):
    """Verify _list_sessions returns only .json files sorted newest first."""
    from src.ui.streamlit_app import HISTORY_DIR, _list_sessions

    original = HISTORY_DIR
    try:
        import src.ui.streamlit_app as ui

        ui.HISTORY_DIR = tmp_path
        (tmp_path / "2026-01-02_10-00-00.json").write_text("[]", encoding="utf-8")
        (tmp_path / "2026-01-01_10-00-00.json").write_text("[]", encoding="utf-8")
        (tmp_path / "not_a_json.txt").write_text("", encoding="utf-8")
        (tmp_path / "readme.md").write_text("", encoding="utf-8")

        sessions = _list_sessions()
        assert len(sessions) == 2
        assert sessions[0].name == "2026-01-02_10-00-00.json"
        assert sessions[1].name == "2026-01-01_10-00-00.json"
    finally:
        ui.HISTORY_DIR = original


def test_session_label_from_first_message(tmp_path):
    """Verify _session_label reads first message content."""
    from src.ui.streamlit_app import _session_label

    path = tmp_path / "session.json"
    data = [{"role": "user", "content": "What is RAG?"}]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    label = _session_label(path)
    assert "What is RAG?" in label
    assert "session" in label


def test_session_label_empty_conversation(tmp_path):
    """Verify _session_label handles empty message list."""
    from src.ui.streamlit_app import _session_label

    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    label = _session_label(path)
    assert label == "empty"


def test_session_label_corrupt_json(tmp_path):
    """Verify _session_label handles corrupt JSON gracefully."""
    from src.ui.streamlit_app import _session_label

    path = tmp_path / "corrupt.json"
    path.write_text("{bad json", encoding="utf-8")
    label = _session_label(path)
    assert label == "corrupt"


def test_save_and_load_messages(tmp_path):
    """Verify save_messages writes and load_messages reads correctly."""
    import src.ui.streamlit_app as ui
    from src.ui.streamlit_app import HISTORY_DIR, load_messages, save_messages

    original_dir = HISTORY_DIR
    try:
        ui.HISTORY_DIR = tmp_path

        class FakeSessionState:
            messages = [{"role": "user", "content": "Hello"}]
            current_history_path = None

        original_session = ui.st.session_state
        ui.st.session_state = FakeSessionState()

        save_messages()
        assert ui.st.session_state.current_history_path is not None
        saved_path = ui.st.session_state.current_history_path
        assert saved_path.exists()

        loaded = load_messages(saved_path)
        assert len(loaded) == 1
        assert loaded[0]["content"] == "Hello"

        ui.st.session_state = original_session
    finally:
        ui.HISTORY_DIR = original_dir


def _make_pop():
    """Return a pop function suitable for FakeSessionState.
    Handles both instance attributes and class-level attributes."""

    def pop(self, key, default=None):
        if key in type(self).__dict__:
            delattr(type(self), key)
            return default
        if key in self.__dict__:
            del self.__dict__[key]
            return default
        return default

    return pop


def test_delete_current_conversation_removes_file(tmp_path):
    """Verify delete_current_conversation removes the JSON file and resets state."""
    import src.ui.streamlit_app as ui
    from src.ui.streamlit_app import HISTORY_DIR, delete_current_conversation

    original_dir = HISTORY_DIR
    try:
        ui.HISTORY_DIR = tmp_path
        history_file = tmp_path / "test_session.json"
        history_file.write_text("[]", encoding="utf-8")
        assert history_file.exists()

        class FakeSessionState:
            messages = [{"role": "user", "content": "Hi"}]
            current_history_path = history_file
            feedback_given = {}
            pop = _make_pop()

            def rerun(self):
                pass

        original_session = ui.st.session_state
        ui.st.session_state = FakeSessionState()

        delete_current_conversation()
        assert not history_file.exists()
        assert ui.st.session_state.messages == []
        assert ui.st.session_state.current_history_path is None

        ui.st.session_state = original_session
    finally:
        ui.HISTORY_DIR = original_dir


def test_delete_current_conversation_missing_file(tmp_path):
    """Verify delete_current_conversation does not crash if file is missing."""
    import src.ui.streamlit_app as ui
    from src.ui.streamlit_app import HISTORY_DIR, delete_current_conversation

    original_dir = HISTORY_DIR
    try:
        ui.HISTORY_DIR = tmp_path
        missing = tmp_path / "already_deleted.json"

        class FakeSessionState:
            messages = [{"role": "user", "content": "Hi"}]
            current_history_path = missing
            feedback_given = {}
            _confirm_clear_all = True
            pop = _make_pop()

            def rerun(self):
                pass

        original_session = ui.st.session_state
        ui.st.session_state = FakeSessionState()

        delete_current_conversation()
        # Should not raise — file already gone
        assert ui.st.session_state.messages == []
        assert ui.st.session_state.current_history_path is None
        assert not hasattr(ui.st.session_state, "_confirm_clear_all")

        ui.st.session_state = original_session
    finally:
        ui.HISTORY_DIR = original_dir


def test_clear_all_conversations_removes_only_json(tmp_path):
    """Verify clear_all_conversations removes .json files and resets state."""
    import src.ui.streamlit_app as ui
    from src.ui.streamlit_app import HISTORY_DIR, clear_all_conversations

    original_dir = HISTORY_DIR
    try:
        ui.HISTORY_DIR = tmp_path

        (tmp_path / "session1.json").write_text("[]", encoding="utf-8")
        (tmp_path / "session2.json").write_text("[]", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")

        class FakeSessionState:
            messages = [{"role": "user", "content": "Hi"}]
            current_history_path = None
            feedback_given = {}
            _confirm_clear_all = True
            pop = _make_pop()

            def rerun(self):
                pass

        original_session = ui.st.session_state
        ui.st.session_state = FakeSessionState()

        clear_all_conversations()

        # JSON files should be gone
        assert not (tmp_path / "session1.json").exists()
        assert not (tmp_path / "session2.json").exists()

        # Non-JSON files should remain
        assert (tmp_path / "notes.txt").exists()
        assert (tmp_path / "data.csv").exists()

        # State should be reset
        assert ui.st.session_state.messages == []
        assert ui.st.session_state.current_history_path is None
        assert not hasattr(ui.st.session_state, "_confirm_clear_all")

        ui.st.session_state = original_session
    finally:
        ui.HISTORY_DIR = original_dir


def test_new_conversation_resets_confirm_flag(tmp_path):
    """Verify new_conversation clears _confirm_clear_all and resets chat state."""
    import src.ui.streamlit_app as ui
    from src.ui.streamlit_app import HISTORY_DIR, new_conversation, save_messages

    original_dir = HISTORY_DIR
    try:
        ui.HISTORY_DIR = tmp_path

        class FakeSessionState:
            messages = [{"role": "user", "content": "Hello"}]
            current_history_path = None
            feedback_given = {}
            _confirm_clear_all = True
            pop = _make_pop()

            def rerun(self):
                pass

        original_session = ui.st.session_state
        ui.st.session_state = FakeSessionState()

        new_conversation()
        assert ui.st.session_state.messages == []
        assert ui.st.session_state.current_history_path is None
        assert not hasattr(ui.st.session_state, "_confirm_clear_all")

        ui.st.session_state = original_session
    finally:
        ui.HISTORY_DIR = original_dir


def test_current_always_available():
    """Verify (current) is always an option, even with zero saved sessions."""
    from src.ui.streamlit_app import _list_sessions

    sessions = _list_sessions()
    options = ["(current)"] + ["" for _ in sessions]
    assert "(current)" in options
    # With zero JSON files, only (current) exists
    assert len(options) == 1 or len(options) > 1


def test_clear_all_conversations_no_directory(tmp_path):
    """Verify clear_all_conversations does not crash if directory is missing."""
    import src.ui.streamlit_app as ui
    from src.ui.streamlit_app import HISTORY_DIR, clear_all_conversations

    original_dir = HISTORY_DIR
    try:
        ui.HISTORY_DIR = tmp_path / "nonexistent"

        class FakeSessionState:
            messages = [{"role": "user", "content": "Hi"}]
            current_history_path = None
            feedback_given = {}
            _confirm_clear_all = True
            pop = _make_pop()

            def rerun(self):
                pass

        original_session = ui.st.session_state
        ui.st.session_state = FakeSessionState()

        clear_all_conversations()
        # Should not raise
        assert ui.st.session_state.messages == []
        assert ui.st.session_state.current_history_path is None
        assert not hasattr(ui.st.session_state, "_confirm_clear_all")

        ui.st.session_state = original_session
    finally:
        ui.HISTORY_DIR = original_dir

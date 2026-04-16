"""Tests for SessionManager."""

import time
from unittest.mock import MagicMock, patch

import pytest

from pydantic_ai_backends import RuntimeConfig, SessionManager


class MockDockerSandbox:
    """Mock DockerSandbox for testing SessionManager."""

    def __init__(
        self,
        runtime: RuntimeConfig | str | None = None,
        session_id: str | None = None,
        idle_timeout: int = 3600,
        volumes: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        self._id = session_id or "test-id"
        self._runtime = runtime
        self._idle_timeout = idle_timeout
        self._last_activity = time.time()
        self._alive = True
        self._volumes = volumes or {}

    @property
    def session_id(self) -> str:
        return self._id

    async def is_alive(self) -> bool:
        return self._alive

    def start(self) -> None:
        self._alive = True

    def stop(self) -> None:
        self._alive = False


class MockCustomSandbox:
    """Mock sandbox for testing custom factory support."""

    def __init__(self, session_id: str) -> None:
        self._id = session_id
        self._last_activity = time.time()
        self._alive = True
        self._started = False

    @property
    def session_id(self) -> str:
        return self._id

    async def is_alive(self) -> bool:
        return self._alive

    def start(self) -> None:
        self._started = True
        self._alive = True

    def stop(self) -> None:
        self._alive = False


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_init_defaults(self):
        """Test default initialization."""
        manager = SessionManager()
        assert manager._default_runtime is None
        assert manager._default_idle_timeout == 3600
        assert manager.session_count == 0
        assert len(manager) == 0

    def test_init_with_runtime(self):
        """Test initialization with default runtime."""
        runtime = RuntimeConfig(name="test")
        manager = SessionManager(default_runtime=runtime)
        assert manager._default_runtime is runtime

    def test_init_with_string_runtime(self):
        """Test initialization with runtime name."""
        manager = SessionManager(default_runtime="python-datascience")
        assert manager._default_runtime == "python-datascience"

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        manager = SessionManager(default_idle_timeout=1800)
        assert manager._default_idle_timeout == 1800

    def test_init_with_factory(self):
        """Test initialization with custom sandbox factory."""

        def factory(sid: str) -> MockCustomSandbox:
            return MockCustomSandbox(sid)

        manager = SessionManager(sandbox_factory=factory)
        assert manager._sandbox_factory is factory

    @pytest.mark.asyncio
    async def test_get_or_create_new_session(self):
        """Test creating a new session."""
        manager = SessionManager()

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox = await manager.get_or_create("user-123")
            assert sandbox.session_id == "user-123"
            assert "user-123" in manager
            assert manager.session_count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_existing_session(self):
        """Test retrieving existing session."""
        manager = SessionManager()

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox1 = await manager.get_or_create("user-123")
            sandbox2 = await manager.get_or_create("user-123")
            assert sandbox1 is sandbox2
            assert manager.session_count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_dead_session_recreates(self):
        """Test that dead sessions are recreated."""
        manager = SessionManager()

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox1 = await manager.get_or_create("user-123")
            sandbox1._alive = False  # type: ignore[attr-defined]  # Mock attribute

            sandbox2 = await manager.get_or_create("user-123")
            assert sandbox1 is not sandbox2
            assert manager.session_count == 1

    @pytest.mark.asyncio
    async def test_release_existing_session(self):
        """Test releasing an existing session."""
        manager = SessionManager()

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            await manager.get_or_create("user-123")
            assert manager.session_count == 1

            result = await manager.release("user-123")
            assert result is True
            assert manager.session_count == 0
            assert "user-123" not in manager

    @pytest.mark.asyncio
    async def test_release_nonexistent_session(self):
        """Test releasing a non-existent session."""
        manager = SessionManager()
        result = await manager.release("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_idle_sessions(self):
        """Test cleaning up idle sessions."""
        manager = SessionManager(default_idle_timeout=10)

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox1 = await manager.get_or_create("user-1")
            sandbox2 = await manager.get_or_create("user-2")

            # Make one session idle
            sandbox1._last_activity = time.time() - 20  # 20 seconds ago
            sandbox2._last_activity = time.time()  # Just now

            cleaned = await manager.cleanup_idle(max_idle=10)
            assert cleaned == 1
            assert manager.session_count == 1
            assert "user-1" not in manager
            assert "user-2" in manager

    @pytest.mark.asyncio
    async def test_cleanup_idle_uses_default_timeout(self):
        """Test cleanup uses default timeout when not specified."""
        manager = SessionManager(default_idle_timeout=5)

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox = await manager.get_or_create("user-1")
            sandbox._last_activity = time.time() - 10  # 10 seconds ago

            cleaned = await manager.cleanup_idle()
            assert cleaned == 1

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutting down all sessions."""
        manager = SessionManager()

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            await manager.get_or_create("user-1")
            await manager.get_or_create("user-2")
            await manager.get_or_create("user-3")
            assert manager.session_count == 3

            count = await manager.shutdown()
            assert count == 3
            assert manager.session_count == 0

    def test_sessions_property(self):
        """Test sessions property returns copy."""
        manager = SessionManager()
        manager._sessions["test"] = MagicMock()

        sessions = manager.sessions
        assert "test" in sessions
        # Verify it's a copy
        sessions["new"] = MagicMock()
        assert "new" not in manager._sessions

    def test_contains(self):
        """Test __contains__ method."""
        manager = SessionManager()
        manager._sessions["test"] = MagicMock()

        assert "test" in manager
        assert "other" not in manager

    def test_len(self):
        """Test __len__ method."""
        manager = SessionManager()
        assert len(manager) == 0

        manager._sessions["a"] = MagicMock()
        manager._sessions["b"] = MagicMock()
        assert len(manager) == 2

    def test_start_cleanup_loop(self):
        """Test starting cleanup loop."""
        manager = SessionManager()
        assert manager._cleanup_task is None

        # We can't actually test the async loop without running it,
        # but we can verify it's created
        with patch("asyncio.create_task") as mock_create_task:
            manager.start_cleanup_loop(interval=60)
            mock_create_task.assert_called_once()

        # Calling again should do nothing
        with patch("asyncio.create_task") as mock_create_task:
            manager._cleanup_task = MagicMock()
            manager.start_cleanup_loop()
            mock_create_task.assert_not_called()

    def test_stop_cleanup_loop(self):
        """Test stopping cleanup loop."""
        manager = SessionManager()
        mock_task = MagicMock()
        manager._cleanup_task = mock_task

        manager.stop_cleanup_loop()
        mock_task.cancel.assert_called_once()
        assert manager._cleanup_task is None

    def test_stop_cleanup_loop_when_not_running(self):
        """Test stopping cleanup loop when not running."""
        manager = SessionManager()
        manager.stop_cleanup_loop()  # Should not raise
        assert manager._cleanup_task is None

    def test_init_with_workspace_root_string(self):
        """Test initialization with workspace_root as string."""
        manager = SessionManager(workspace_root="/tmp/sessions")
        assert manager._workspace_root is not None
        assert str(manager._workspace_root) == "/tmp/sessions"

    def test_init_with_workspace_root_path(self):
        """Test initialization with workspace_root as Path."""
        from pathlib import Path

        manager = SessionManager(workspace_root=Path("/tmp/sessions"))
        assert manager._workspace_root is not None
        assert str(manager._workspace_root) == "/tmp/sessions"

    def test_init_without_workspace_root(self):
        """Test initialization without workspace_root."""
        manager = SessionManager()
        assert manager._workspace_root is None

    @pytest.mark.asyncio
    async def test_get_or_create_with_workspace_root(self, tmp_path):
        """Test that workspace_root creates directories and passes volumes."""
        manager = SessionManager(workspace_root=tmp_path)

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox = await manager.get_or_create("user-123")

            # Check that directory was created
            expected_dir = tmp_path / "user-123" / "workspace"
            assert expected_dir.exists()
            assert expected_dir.is_dir()

            # Check that volumes were passed to sandbox
            assert sandbox._volumes is not None
            assert str(expected_dir.resolve()) in sandbox._volumes
            assert sandbox._volumes[str(expected_dir.resolve())] == "/workspace"

    @pytest.mark.asyncio
    async def test_get_or_create_without_workspace_root_no_volumes(self):
        """Test that without workspace_root, no volumes are set."""
        manager = SessionManager()

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox = await manager.get_or_create("user-123")
            assert sandbox._volumes == {}

    @pytest.mark.asyncio
    async def test_get_or_create_multiple_sessions_separate_dirs(self, tmp_path):
        """Test that multiple sessions get separate workspace directories."""
        manager = SessionManager(workspace_root=tmp_path)

        with patch("pydantic_ai_backends.backends.docker.sandbox.DockerSandbox", MockDockerSandbox):
            sandbox1 = await manager.get_or_create("user-1")
            sandbox2 = await manager.get_or_create("user-2")

            # Check separate directories
            dir1 = tmp_path / "user-1" / "workspace"
            dir2 = tmp_path / "user-2" / "workspace"

            assert dir1.exists()
            assert dir2.exists()
            assert dir1 != dir2

            # Check separate volumes
            assert str(dir1.resolve()) in sandbox1._volumes
            assert str(dir2.resolve()) in sandbox2._volumes


class TestSessionManagerWithFactory:
    """Tests for SessionManager with custom sandbox_factory."""

    @pytest.mark.asyncio
    async def test_factory_called_with_session_id(self):
        """Test that factory receives the session_id."""
        created_ids: list[str] = []

        def factory(session_id: str) -> MockCustomSandbox:
            created_ids.append(session_id)
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        await manager.get_or_create("user-42")

        assert created_ids == ["user-42"]

    @pytest.mark.asyncio
    async def test_factory_sandbox_started(self):
        """Test that factory-created sandboxes get start() called."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        sandbox = await manager.get_or_create("user-1")
        assert sandbox._started is True

    @pytest.mark.asyncio
    async def test_factory_sandbox_reused_when_alive(self):
        """Test that alive factory sandboxes are reused."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        s1 = await manager.get_or_create("user-1")
        s2 = await manager.get_or_create("user-1")
        assert s1 is s2

    @pytest.mark.asyncio
    async def test_factory_sandbox_recreated_when_dead(self):
        """Test that dead factory sandboxes are recreated."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        s1 = await manager.get_or_create("user-1")
        s1._alive = False

        s2 = await manager.get_or_create("user-1")
        assert s1 is not s2
        assert s2._started is True

    @pytest.mark.asyncio
    async def test_factory_release_stops_sandbox(self):
        """Test that releasing a factory sandbox calls stop()."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        sandbox = await manager.get_or_create("user-1")
        assert sandbox._alive is True

        await manager.release("user-1")
        assert sandbox._alive is False
        assert "user-1" not in manager

    @pytest.mark.asyncio
    async def test_factory_cleanup_idle(self):
        """Test idle cleanup with factory sandboxes."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(
            sandbox_factory=factory,
            default_idle_timeout=10,
        )
        s1 = await manager.get_or_create("user-1")
        s2 = await manager.get_or_create("user-2")

        s1._last_activity = time.time() - 20  # idle
        s2._last_activity = time.time()  # active

        cleaned = await manager.cleanup_idle()
        assert cleaned == 1
        assert "user-1" not in manager
        assert "user-2" in manager

    @pytest.mark.asyncio
    async def test_factory_shutdown(self):
        """Test shutdown with factory sandboxes."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        await manager.get_or_create("a")
        await manager.get_or_create("b")

        count = await manager.shutdown()
        assert count == 2
        assert manager.session_count == 0

    @pytest.mark.asyncio
    async def test_factory_ignores_runtime_param(self):
        """Test that runtime param is ignored when using factory."""
        factory_calls: list[str] = []

        def factory(session_id: str) -> MockCustomSandbox:
            factory_calls.append(session_id)
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        # Pass runtime — should be ignored (factory doesn't receive it)
        await manager.get_or_create("user-1", runtime="python-datascience")
        assert factory_calls == ["user-1"]

    @pytest.mark.asyncio
    async def test_factory_ignores_workspace_root(self, tmp_path):
        """Test that workspace_root doesn't affect factory-created sandboxes."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(
            sandbox_factory=factory,
            workspace_root=tmp_path,
        )
        await manager.get_or_create("user-1")

        # workspace_root should NOT create directories for factory sandboxes
        assert not (tmp_path / "user-1").exists()

    @pytest.mark.asyncio
    async def test_factory_activity_updated_on_reuse(self):
        """Test that _last_activity is updated when reusing a session."""

        def factory(session_id: str) -> MockCustomSandbox:
            return MockCustomSandbox(session_id)

        manager = SessionManager(sandbox_factory=factory)
        sandbox = await manager.get_or_create("user-1")
        sandbox._last_activity = time.time() - 100  # Simulate old activity

        before = sandbox._last_activity
        await manager.get_or_create("user-1")
        assert sandbox._last_activity > before

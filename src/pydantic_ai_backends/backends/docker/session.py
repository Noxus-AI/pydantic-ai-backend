"""Session management for sandbox backends.

Provides session management for multi-user applications,
allowing multiple users to have their own isolated sandboxes
(Docker, Daytona, or any custom sandbox type).

Example with default Docker backend:
    ```python
    from pydantic_ai_backends import SessionManager

    manager = SessionManager(default_runtime="python-datascience")
    sandbox = await manager.get_or_create("user-123")
    result = sandbox.execute("python script.py")
    await manager.release("user-123")
    ```

Example with custom factory:
    ```python
    from pydantic_ai_backends import SessionManager, DaytonaSandbox

    def daytona_factory(session_id: str) -> DaytonaSandbox:
        return DaytonaSandbox(sandbox_id=session_id)

    manager = SessionManager(sandbox_factory=daytona_factory)
    sandbox = await manager.get_or_create("user-123")
    ```
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai_backends.types import RuntimeConfig

#: Factory callable: receives ``session_id`` and returns a sandbox instance.
SandboxFactory = Callable[[str], Any]


class SessionManager:
    """Manages user sessions and their sandbox containers.

    This class provides a way to manage multiple sandbox instances
    for different user sessions. It handles:

    - Creating new sandboxes for new sessions
    - Reusing existing sandboxes for returning sessions
    - Cleaning up idle sessions automatically

    By default, sandboxes are created using :class:`DockerSandbox`.
    Pass a ``sandbox_factory`` callable to use a different backend
    (e.g. :class:`DaytonaSandbox` or a custom implementation).

    Example:
        ```python
        from pydantic_ai_backends import SessionManager

        # Docker (default)
        manager = SessionManager(default_runtime="python-datascience")

        # Custom factory (e.g. Daytona)
        manager = SessionManager(sandbox_factory=my_factory)

        sandbox = await manager.get_or_create("user-123")
        cleaned = await manager.cleanup_idle(max_idle=1800)
        ```
    """

    def __init__(
        self,
        sandbox_factory: SandboxFactory | None = None,
        default_runtime: RuntimeConfig | str | None = None,
        default_idle_timeout: int = 3600,
        workspace_root: str | Path | None = None,
    ):
        """Initialize the session manager.

        Args:
            sandbox_factory: Callable that receives a ``session_id`` string and
                returns a sandbox instance with ``start()``, ``stop()``,
                ``is_alive()`` methods and a ``_last_activity`` attribute.
                When ``None``, defaults to creating :class:`DockerSandbox` instances.
            default_runtime: Default RuntimeConfig or name for new Docker sandboxes.
                Only used when ``sandbox_factory`` is ``None``.
            default_idle_timeout: Default idle timeout in seconds (default: 1 hour).
            workspace_root: Root directory for persistent session storage.
                Only used when ``sandbox_factory`` is ``None``.
                If set, creates ``{workspace_root}/{session_id}/workspace``
                and mounts it as a Docker volume.
        """
        self._sessions: dict[str, Any] = {}
        self._sandbox_factory = sandbox_factory
        self._default_runtime = default_runtime
        self._default_idle_timeout = default_idle_timeout
        self._cleanup_task: asyncio.Task[None] | None = None
        self._workspace_root = Path(workspace_root) if workspace_root else None

    @property
    def sessions(self) -> dict[str, Any]:
        """Active sessions dictionary (read-only copy)."""
        return dict(self._sessions)

    @property
    def session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    async def get_or_create(
        self,
        session_id: str,
        runtime: RuntimeConfig | str | None = None,
    ) -> Any:
        """Get an existing sandbox or create a new one.

        If a sandbox exists for the ``session_id`` and is still alive,
        it will be returned. Otherwise, a new sandbox will be created.

        Args:
            session_id: Unique identifier for the session.
            runtime: RuntimeConfig or name to use. Only applies when using
                the default Docker factory (ignored with custom ``sandbox_factory``).

        Returns:
            Sandbox instance for the session.
        """
        # Check for existing session
        if session_id in self._sessions:
            sandbox = self._sessions[session_id]
            if await sandbox.is_alive():
                sandbox._last_activity = time.time()
                return sandbox
            # Container died, remove from cache
            del self._sessions[session_id]

        # Create via factory or default Docker path
        if self._sandbox_factory is not None:
            sandbox = self._sandbox_factory(session_id)
        else:
            sandbox = self._create_docker_sandbox(session_id, runtime)

        sandbox.start()
        self._sessions[session_id] = sandbox
        return sandbox

    def _create_docker_sandbox(
        self,
        session_id: str,
        runtime: RuntimeConfig | str | None = None,
    ) -> Any:
        """Create a DockerSandbox with the legacy configuration.

        This is the default path when no ``sandbox_factory`` is provided.
        """
        from pydantic_ai_backends.backends.docker.sandbox import DockerSandbox

        # Prepare volumes for persistent storage
        volumes: dict[str, str] | None = None
        if self._workspace_root:
            session_workspace = self._workspace_root / session_id / "workspace"
            session_workspace.mkdir(parents=True, exist_ok=True)
            volumes = {str(session_workspace.resolve()): "/workspace"}

        effective_runtime = runtime or self._default_runtime
        return DockerSandbox(
            runtime=effective_runtime,
            session_id=session_id,
            idle_timeout=self._default_idle_timeout,
            volumes=volumes,
        )

    async def release(self, session_id: str) -> bool:
        """Release a session and stop its sandbox.

        Args:
            session_id: Session identifier to release.

        Returns:
            True if session was found and released, False otherwise.
        """
        if session_id not in self._sessions:
            return False

        sandbox = self._sessions.pop(session_id)
        sandbox.stop()
        return True

    async def cleanup_idle(self, max_idle: int | None = None) -> int:
        """Clean up idle sessions.

        Removes and stops sandboxes that have been idle for longer than
        the specified time.

        Args:
            max_idle: Maximum idle time in seconds. Uses default if not specified.

        Returns:
            Number of sessions cleaned up.
        """
        max_idle = max_idle if max_idle is not None else self._default_idle_timeout
        now = time.time()
        to_remove: list[str] = []

        for session_id, sandbox in self._sessions.items():
            if now - sandbox._last_activity > max_idle:
                to_remove.append(session_id)

        for session_id in to_remove:
            await self.release(session_id)

        return len(to_remove)

    def start_cleanup_loop(self, interval: int = 300) -> None:
        """Start background cleanup loop.

        Periodically cleans up idle sessions.

        Args:
            interval: Cleanup interval in seconds (default: 5 minutes).
        """
        if self._cleanup_task is not None:
            return  # Already running

        async def _loop() -> None:  # pragma: no cover
            while True:
                await asyncio.sleep(interval)
                await self.cleanup_idle()

        self._cleanup_task = asyncio.create_task(_loop())

    def stop_cleanup_loop(self) -> None:
        """Stop the background cleanup loop."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def shutdown(self) -> int:
        """Shutdown all sessions and stop cleanup loop.

        Returns:
            Number of sessions that were stopped.
        """
        self.stop_cleanup_loop()

        count = len(self._sessions)
        session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            await self.release(session_id)

        return count

    def __contains__(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    def __len__(self) -> int:
        """Return number of active sessions."""
        return len(self._sessions)

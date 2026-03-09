"""
Transport-aware OAuth callback handling.

In streamable-http mode: Uses the existing FastAPI server
In stdio mode: Starts a minimal HTTP server just for OAuth callbacks
"""

import asyncio
import fcntl
import logging
import os
import threading
import time
import socket
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from urllib.parse import urlparse

from auth.scopes import SCOPES, get_current_scopes  # noqa
from auth.oauth_responses import (
    create_error_response,
    create_success_response,
    create_server_error_response,
)
from auth.google_auth import handle_auth_callback, check_client_secrets
from auth.oauth_config import get_oauth_redirect_uri

logger = logging.getLogger(__name__)

# Lock file protocol — coordinates which MCP instance owns the OAuth server
LOCK_DIR = os.path.expanduser("~/.google_workspace_mcp")
LOCK_FILE = os.path.join(LOCK_DIR, "oauth_server.lock")
PID_FILE = os.path.join(LOCK_DIR, "oauth_server.pid")


class MinimalOAuthServer:
    """
    Minimal HTTP server for OAuth callbacks in stdio mode.
    Only starts when needed and uses the same port (8000) as streamable-http mode.
    """

    def __init__(self, port: int = 8000, base_uri: str = "http://localhost"):
        self.port = port
        self.base_uri = base_uri
        self.app = FastAPI()
        self.server = None
        self.server_thread = None
        self.is_running = False

        # Lock state
        self._lock_fd = None
        self._owns_lock = False
        self._reusing_existing = False

        # Idle auto-shutdown state
        self._last_request_time: float = time.time()
        self._idle_timeout: int = 60  # seconds
        self._idle_check_interval: int = 15  # seconds
        self._idle_timer_thread: Optional[threading.Thread] = None
        self._shutting_down: bool = False

        # Setup the callback route
        self._setup_callback_route()
        # Setup attachment serving route
        self._setup_attachment_route()
        # Setup activity tracking middleware for idle auto-shutdown
        self._setup_activity_tracking()

    def _setup_callback_route(self):
        """Setup the OAuth callback route."""

        @self.app.get("/oauth2callback")
        async def oauth_callback(request: Request):
            """Handle OAuth callback - same logic as in core/server.py"""
            code = request.query_params.get("code")
            error = request.query_params.get("error")

            if error:
                error_message = (
                    f"Authentication failed: Google returned an error: {error}."
                )
                logger.error(error_message)
                return create_error_response(error_message)

            if not code:
                error_message = (
                    "Authentication failed: No authorization code received from Google."
                )
                logger.error(error_message)
                return create_error_response(error_message)

            try:
                # Check if we have credentials available (environment variables or file)
                error_message = check_client_secrets()
                if error_message:
                    return create_server_error_response(error_message)

                logger.info(
                    "OAuth callback: Received authorization code. Attempting to exchange for tokens."
                )

                # Session ID tracking removed - not needed

                # Exchange code for credentials
                redirect_uri = get_oauth_redirect_uri()
                verified_user_id, credentials = handle_auth_callback(
                    scopes=get_current_scopes(),
                    authorization_response=str(request.url),
                    redirect_uri=redirect_uri,
                    session_id=None,
                )

                logger.info(
                    f"OAuth callback: Successfully authenticated user: {verified_user_id}."
                )

                # Return success page using shared template
                return create_success_response(verified_user_id)

            except Exception as e:
                error_message_detail = f"Error processing OAuth callback: {str(e)}"
                logger.error(error_message_detail, exc_info=True)
                return create_server_error_response(str(e))

    def _setup_attachment_route(self):
        """Setup the attachment serving route."""
        from core.attachment_storage import get_attachment_storage

        @self.app.get("/attachments/{file_id}")
        async def serve_attachment(file_id: str, request: Request):
            """Serve a stored attachment file."""
            storage = get_attachment_storage()
            metadata = storage.get_attachment_metadata(file_id)

            if not metadata:
                return JSONResponse(
                    {"error": "Attachment not found or expired"}, status_code=404
                )

            file_path = storage.get_attachment_path(file_id)
            if not file_path:
                return JSONResponse(
                    {"error": "Attachment file not found"}, status_code=404
                )

            return FileResponse(
                path=str(file_path),
                filename=metadata["filename"],
                media_type=metadata["mime_type"],
            )

    def _setup_activity_tracking(self):
        """Setup middleware to reset idle timer on every request."""

        @self.app.middleware("http")
        async def track_activity(request: Request, call_next):
            self._last_request_time = time.time()
            return await call_next(request)

    def _idle_check_loop(self):
        """Background thread that checks for idle timeout."""
        while not self._shutting_down:
            time.sleep(self._idle_check_interval)
            if self._shutting_down:
                break
            elapsed = time.time() - self._last_request_time
            if elapsed >= self._idle_timeout:
                logger.info(
                    f"OAuth callback server idle for {elapsed:.0f}s — shutting down"
                )
                self.stop()
                break

    def _start_idle_timer(self):
        """Start the idle check background thread."""
        self._shutting_down = False
        self._last_request_time = time.time()
        self._idle_timer_thread = threading.Thread(
            target=self._idle_check_loop, daemon=True
        )
        self._idle_timer_thread.start()
        logger.info(
            f"Idle auto-shutdown timer started ({self._idle_timeout}s timeout)"
        )

    def _acquire_lock(self) -> bool:
        """
        Try to acquire the OAuth server lock.

        Returns True if we acquired the lock (self._owns_lock) or another
        live instance owns it (self._reusing_existing). Returns False only
        when a stale lock was cleaned up and the caller should retry.
        """
        os.makedirs(LOCK_DIR, exist_ok=True)

        self._lock_fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # We acquired the lock — write our PID
            with open(PID_FILE, "w") as f:
                f.write(str(os.getpid()))
            self._owns_lock = True
            self._reusing_existing = False
            logger.info(f"Acquired OAuth server lock (PID {os.getpid()})")
            return True
        except (IOError, OSError):
            # Another process holds the lock — check if it's still alive
            self._lock_fd.close()
            self._lock_fd = None
            return self._check_existing_owner()

    def _check_existing_owner(self) -> bool:
        """
        Check whether the process that holds the lock is still alive.

        Returns True if alive (sets self._reusing_existing).
        Returns False if stale (cleans up files so caller can retry).
        """
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # Signal 0: check if process exists (doesn't kill)
            logger.info(
                f"Another MCP instance (PID {pid}) owns the OAuth server — reusing"
            )
            self._reusing_existing = True
            self._owns_lock = False
            return True
        except (FileNotFoundError, ValueError, ProcessLookupError):
            # PID file missing, corrupted, or process dead — stale lock
            logger.info("Stale OAuth server lock detected — cleaning up")
            self._cleanup_lock_files()
            return False
        except PermissionError:
            # Process exists but owned by another user — treat as alive
            logger.info(
                "OAuth server lock held by process owned by another user — reusing"
            )
            self._reusing_existing = True
            self._owns_lock = False
            return True

    def _cleanup_lock_files(self):
        """Remove lock and PID files safely."""
        for path in (PID_FILE, LOCK_FILE):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        if self._lock_fd is not None:
            try:
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None
        self._owns_lock = False
        self._reusing_existing = False

    def _release_lock(self):
        """Release the lock and clean up lock/PID files."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except Exception:
                pass
        self._cleanup_lock_files()

    def start(self) -> tuple[bool, str]:
        """
        Start the minimal OAuth server, or detect that another instance owns it.

        Uses a cooperative file-lock protocol so multiple MCP instances
        (one per VS Code workspace) coordinate who binds port 8000.

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        if self.is_running:
            logger.info("Minimal OAuth server is already running")
            return True, ""

        # Extract hostname from base_uri (e.g., "http://localhost" -> "localhost")
        try:
            parsed_uri = urlparse(self.base_uri)
            hostname = parsed_uri.hostname or "localhost"
        except Exception:
            hostname = "localhost"

        # Try to acquire the cooperative lock (max 2 attempts for stale recovery)
        for attempt in range(2):
            acquired = self._acquire_lock()
            if acquired:
                break
            # _acquire_lock returned False means stale lock was cleaned up — retry
            logger.info(f"Retrying lock acquisition (attempt {attempt + 2}/2)")
        else:
            error_msg = "Failed to acquire OAuth server lock after 2 attempts"
            logger.error(error_msg)
            return False, error_msg

        # If another live instance owns the server, we're done
        if self._reusing_existing:
            self.is_running = True
            logger.info(
                f"Reusing OAuth server owned by another MCP instance on {hostname}:{self.port}"
            )
            return True, ""

        # We own the lock — bind port and start serving
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((hostname, self.port))
        except OSError:
            error_msg = (
                f"Port {self.port} is already in use on {hostname}. "
                "Cannot start minimal OAuth server."
            )
            logger.error(error_msg)
            self._release_lock()
            return False, error_msg

        def run_server():
            """Run the server in a separate thread."""
            try:
                config = uvicorn.Config(
                    self.app,
                    host=hostname,
                    port=self.port,
                    log_level="warning",
                    access_log=False,
                )
                self.server = uvicorn.Server(config)
                asyncio.run(self.server.serve())

            except Exception as e:
                logger.error(f"Minimal OAuth server error: {e}", exc_info=True)
                self.is_running = False

        # Start server in background thread
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to start
        max_wait = 3.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex((hostname, self.port))
                    if result == 0:
                        self.is_running = True
                        logger.info(
                            f"Minimal OAuth server started on {hostname}:{self.port}"
                        )
                        self._start_idle_timer()
                        return True, ""
            except Exception:
                pass
            time.sleep(0.1)

        error_msg = (
            f"Failed to start minimal OAuth server on {hostname}:{self.port} "
            f"- server did not respond within {max_wait}s"
        )
        logger.error(error_msg)
        self._release_lock()
        return False, error_msg

    def stop(self):
        """Stop the minimal OAuth server and release the lock if we own it."""
        if not self.is_running:
            return

        # Signal idle timer thread to stop first
        self._shutting_down = True

        try:
            if self._owns_lock:
                # We own the server — shut it down and release the lock
                if self.server:
                    if hasattr(self.server, "should_exit"):
                        self.server.should_exit = True

                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=3.0)

                # Wait for idle timer thread to finish
                if (
                    self._idle_timer_thread
                    and self._idle_timer_thread.is_alive()
                    and self._idle_timer_thread is not threading.current_thread()
                ):
                    self._idle_timer_thread.join(timeout=2.0)

                self._release_lock()
                logger.info("Minimal OAuth server stopped and lock released")
            elif self._reusing_existing:
                # We don't own the server — just clear our state
                logger.info("Detaching from OAuth server owned by another instance")
            else:
                logger.info("Minimal OAuth server stopped (no lock held)")

            self.is_running = False

        except Exception as e:
            logger.error(f"Error stopping minimal OAuth server: {e}", exc_info=True)


# Global instance for stdio mode
_minimal_oauth_server: Optional[MinimalOAuthServer] = None


def ensure_oauth_callback_available(
    transport_mode: str = "stdio", port: int = 8000, base_uri: str = "http://localhost"
) -> tuple[bool, str]:
    """
    Ensure OAuth callback endpoint is available for the given transport mode.

    For streamable-http: Assumes the main server is already running
    For stdio: Starts a minimal server if needed

    Args:
        transport_mode: "stdio" or "streamable-http"
        port: Port number (default 8000)
        base_uri: Base URI (default "http://localhost")

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    global _minimal_oauth_server

    if transport_mode == "streamable-http":
        # In streamable-http mode, the main FastAPI server should handle callbacks
        logger.debug(
            "Using existing FastAPI server for OAuth callbacks (streamable-http mode)"
        )
        return True, ""

    elif transport_mode == "stdio":
        # In stdio mode, start minimal server if not already running
        # If instance exists but was auto-shutdown, create a fresh one
        if _minimal_oauth_server is not None and _minimal_oauth_server._shutting_down:
            logger.info(
                "Previous OAuth server instance was auto-shutdown — creating fresh instance"
            )
            _minimal_oauth_server = None

        if _minimal_oauth_server is None:
            logger.info(f"Creating minimal OAuth server instance for {base_uri}:{port}")
            _minimal_oauth_server = MinimalOAuthServer(port, base_uri)

        if not _minimal_oauth_server.is_running:
            logger.info("Starting minimal OAuth server for stdio mode")
            success, error_msg = _minimal_oauth_server.start()
            if success:
                if _minimal_oauth_server._reusing_existing:
                    logger.info(
                        f"OAuth callback available via another MCP instance on {base_uri}:{port}"
                    )
                else:
                    logger.info(
                        f"Minimal OAuth server successfully started on {base_uri}:{port}"
                    )
                return True, ""
            else:
                logger.error(
                    f"Failed to start minimal OAuth server on {base_uri}:{port}: {error_msg}"
                )
                return False, error_msg
        else:
            logger.info("Minimal OAuth server is already running")
            return True, ""

    else:
        error_msg = f"Unknown transport mode: {transport_mode}"
        logger.error(error_msg)
        return False, error_msg


def cleanup_oauth_callback_server():
    """Clean up the minimal OAuth server if it was started."""
    global _minimal_oauth_server
    if _minimal_oauth_server:
        _minimal_oauth_server.stop()
        _minimal_oauth_server = None

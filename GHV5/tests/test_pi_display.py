"""Tests for ghv5.pi_display — layout, state, demo thread, queue drain."""
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ghv5.config import CELL_LABELS, PI_SCREEN_SIZE

# pygame may not be installed in all test environments
pygame = pytest.importorskip("pygame")

from ghv5.pi_display import GridDisplay, DemoThread  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_pygame_display():
    """Prevent actual pygame window creation during tests."""
    with patch("pygame.init"), \
         patch("pygame.display.set_mode", return_value=MagicMock()), \
         patch("pygame.display.set_caption"), \
         patch("pygame.font.SysFont", return_value=MagicMock()), \
         patch("pygame.quit"):
        yield


def _make_display():
    return GridDisplay(screen_size=PI_SCREEN_SIZE, fullscreen=False)


# ---------------------------------------------------------------------------
# Layout tests
# ---------------------------------------------------------------------------
class TestLayout:

    def test_nine_cell_rects(self):
        d = _make_display()
        assert len(d._cell_rects) == 9

    def test_cells_within_screen(self):
        d = _make_display()
        sw, sh = PI_SCREEN_SIZE
        for (r, c), rect in d._cell_rects.items():
            assert rect.left >= 0, f"r{r}c{c} left out of bounds"
            assert rect.top >= 0, f"r{r}c{c} top out of bounds"
            assert rect.right <= sw, f"r{r}c{c} right out of bounds"
            assert rect.bottom <= sh, f"r{r}c{c} bottom out of bounds"

    def test_cells_non_overlapping(self):
        d = _make_display()
        rects = list(d._cell_rects.values())
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert not a.colliderect(b), f"{a} overlaps {b}"

    def test_four_shouter_positions(self):
        d = _make_display()
        assert len(d._shouter_positions) == 4
        for sid in (1, 2, 3, 4):
            assert sid in d._shouter_positions


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------
class TestState:

    def test_update_sets_current_cell(self):
        d = _make_display()
        d.update("r1c2", 0.85)
        assert d._current_cell == "r1c2"
        assert d._confidence == 0.85
        assert d._last_update_time is not None

    def test_set_status(self):
        d = _make_display()
        d.set_status("Connected")
        assert d._status_msg == "Connected"

    def test_initial_state(self):
        d = _make_display()
        assert d._current_cell is None
        assert d._confidence == 0.0
        assert d._status_msg == "Waiting..."


# ---------------------------------------------------------------------------
# Demo thread
# ---------------------------------------------------------------------------
class TestDemoThread:

    def test_produces_valid_cells(self):
        q = queue.Queue()
        stop = threading.Event()
        t = DemoThread(q, stop)
        t.start()
        try:
            items = []
            deadline = time.time() + 6
            while len(items) < 3 and time.time() < deadline:
                try:
                    item = q.get(timeout=0.5)
                    if item.get("type") == "prediction":
                        items.append(item)
                except queue.Empty:
                    pass
            assert len(items) >= 2, f"Expected >=2 predictions, got {len(items)}"
            for item in items:
                assert item["cell"] in CELL_LABELS
                assert 0.0 <= item["confidence"] <= 1.0
        finally:
            stop.set()
            t.join(timeout=2.0)

    def test_stop_event_halts_thread(self):
        q = queue.Queue()
        stop = threading.Event()
        t = DemoThread(q, stop)
        t.start()
        time.sleep(0.3)
        stop.set()
        t.join(timeout=3.0)
        assert not t.is_alive()


# ---------------------------------------------------------------------------
# Queue drain pattern
# ---------------------------------------------------------------------------
class TestQueueDrain:

    def test_latest_only(self):
        """Simulates the main loop queue drain — only the last prediction matters."""
        q = queue.Queue()
        for i in range(5):
            q.put({
                "type": "prediction",
                "cell": CELL_LABELS[i],
                "confidence": 0.5 + i * 0.1,
            })

        # Drain like main loop does
        latest = None
        try:
            while True:
                item = q.get_nowait()
                if item.get("type") == "prediction":
                    latest = item
        except queue.Empty:
            pass

        assert latest is not None
        assert latest["cell"] == CELL_LABELS[4]
        assert latest["confidence"] == pytest.approx(0.9)

    def test_status_messages_not_overwritten(self):
        """Status messages should update display even when mixed with predictions."""
        q = queue.Queue()
        q.put({"type": "status", "msg": "Connected"})
        q.put({"type": "prediction", "cell": "r0c0", "confidence": 0.8})

        latest_pred = None
        statuses = []
        try:
            while True:
                item = q.get_nowait()
                if item.get("type") == "prediction":
                    latest_pred = item
                elif item.get("type") == "status":
                    statuses.append(item["msg"])
        except queue.Empty:
            pass

        assert latest_pred["cell"] == "r0c0"
        assert "Connected" in statuses



from __future__ import annotations

from typing import Any, Dict

from pythonosc.udp_client import SimpleUDPClient


class OscClient:
    """
    Simple OSC client to talk to SuperCollider on localhost.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 57120) -> None:
        self.host = host
        self.port = port
        self._client = SimpleUDPClient(self.host, self.port)

    def send_anomaly(self, band_name: str, center_freq_hz: float, payload: Dict[str, Any]) -> None:
        addr = "/seismoEM/anomaly"
        msg = {
            "band": band_name,
            "center_freq_hz": center_freq_hz,
            **payload,
        }
        # Send as separate arguments to keep it SuperCollider-friendly
        self._client.send_message(
            addr,
            [
                msg.get("band"),
                float(msg.get("center_freq_hz", 0.0)),
                float(msg.get("power_db", 0.0)),
                float(msg.get("delta_db", 0.0)),
            ],
        )



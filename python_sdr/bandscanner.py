from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .anomaly_engine import AnomalyEngine
from .dsp_core import compute_power_spectrum
from .osc_client import OscClient
from .rtl_device import RtlDevice


class BandScanner:
    """
    Simple stepping band scanner for one RTL-SDR device.
    """

    def __init__(
        self,
        device: RtlDevice,
        band_config_path: Path,
        antenna_key: str,
        osc_client: OscClient | None = None,
    ) -> None:
        self.device = device
        self.osc_client = osc_client or OscClient()
        self.antenna_key = antenna_key
        self.anomaly_engine = AnomalyEngine()

        with band_config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.bands: List[Dict] = cfg.get(antenna_key, [])

    def iter_band_points(self) -> Iterable[Dict]:
        for band in self.bands:
            start = band["start_hz"]
            stop = band["stop_hz"]
            step = band["step_hz"]
            name = band["name"]
            freq = start
            while freq <= stop:
                yield {"band_name": name, "center_freq_hz": freq}
                freq += step

    def scan_once(self) -> None:
        for point in self.iter_band_points():
            band_name = point["band_name"]
            center_freq_hz = point["center_freq_hz"]
            self.device.set_center_frequency(center_freq_hz)
            samples = self.device.read_samples()
            freqs, power_db = compute_power_spectrum(
                samples, sample_rate_hz=self.device.config.sample_rate_hz
            )
            anomalies = self.anomaly_engine.update_and_detect(
                band_name, freqs, power_db
            )
            for freq_hz, delta_db in anomalies:
                self.osc_client.send_anomaly(
                    band_name,
                    center_freq_hz,
                    {"power_db": float(np.max(power_db)), "delta_db": delta_db},
                )



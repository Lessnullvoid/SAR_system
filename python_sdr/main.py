from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bandscanner import BandScanner
from .logger import setup_logging
from .osc_client import OscClient
from .rtl_device import RtlDevice, RtlDeviceConfig


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "python_app" / "config"


def load_antenna_config(antenna_key: str) -> dict:
    cfg_path = CONFIG_DIR / "antennas.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if antenna_key not in cfg:
        raise KeyError(f"Antenna '{antenna_key}' not found in antennas.json")
    return cfg[antenna_key]


def run_radio_mode(
    device_cfg: RtlDeviceConfig,
    center_freq_hz: float,
    demod: str,
) -> None:
    """
    Simple "classic radio" state.

    For now this just keeps the tuner locked on a given frequency and
    prints a rough RF level so that higher-level GUI components can
    attach for spectrum / audio later.
    """
    with RtlDevice(device_cfg) as dev:
        dev.set_center_frequency(center_freq_hz)
        print(
            f"Radio mode: {demod.upper()} at {center_freq_hz/1e6:.6f} MHz "
            "(Ctrl-C to quit)"
        )
        try:
            while True:
                samples = dev.read_samples()
                level = 20.0 * np.log10(np.mean(np.abs(samples)) + 1e-12)
                print(f"RF level: {level:6.1f} dBFS", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\nStopping radio mode.")


def run_seismo_em_mode(
    antenna_key: str,
    device_cfg: RtlDeviceConfig,
    scan_sleep: float,
    once: bool,
) -> None:
    osc_client = OscClient()
    band_config_path = CONFIG_DIR / "freq_bands.json"

    with RtlDevice(device_cfg) as dev:
        scanner = BandScanner(
            device=dev,
            band_config_path=band_config_path,
            antenna_key=antenna_key,
            osc_client=osc_client,
        )

        try:
            if once:
                scanner.scan_once()
            else:
                while True:
                    scanner.scan_once()
                    time.sleep(scan_sleep)
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seismo-EM SDR backend.\n"
            "Two main states:\n"
            "  1) 'radio'     – classic manual tuning (AM/FM-style)\n"
            "  2) 'seismo-em' – autonomous band scanner + anomaly detector"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["radio", "seismo-em"],
        default="radio",
        help="Operating mode: classic radio vs Seismo-EM scanner.",
    )
    parser.add_argument(
        "--antenna",
        choices=["loop_antenna", "fm_broadcast", "discone"],
        default="loop_antenna",
        help="Which antenna / RTL-SDR device configuration to use.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform a single scan pass over all bands, then exit.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay in seconds between full scan passes (when not using --once).",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=100.0,
        help="Tuning frequency in MHz for 'radio' mode.",
    )
    parser.add_argument(
        "--demod",
        choices=["am", "fm"],
        default="fm",
        help="Demodulation type for 'radio' mode (future use; for now informational).",
    )
    args = parser.parse_args()

    setup_logging()

    antenna_cfg = load_antenna_config(args.antenna)

    # Center frequency here is only an initial value; BandScanner will step it
    # in Seismo-EM mode, and in radio mode we override with --freq-mhz.
    device_cfg = RtlDeviceConfig(
        device_index=int(antenna_cfg["device_index"]),
        center_freq_hz=1_000_000.0,
        sample_rate_hz=float(antenna_cfg["default_sample_rate"]),
        gain_db=float(antenna_cfg["default_gain_db"]),
        direct_sampling_q=bool(antenna_cfg.get("use_direct_sampling_q", False)),
    )

    if args.mode == "radio":
        center_hz = args.freq_mhz * 1e6
        run_radio_mode(device_cfg, center_hz, demod=args.demod)
    else:
        run_seismo_em_mode(
            antenna_key=args.antenna,
            device_cfg=device_cfg,
            scan_sleep=args.sleep,
            once=args.once,
        )


if __name__ == "__main__":
    main()



"""
S.A.R — SDR application package.

Entry point: python -m python_app.gui_main

Provides:
- RTL-SDR device management (rtl_device)
- DSP pipeline: FFT, filtering, decimation (dsp_core)
- FM / AM / SSB demodulators with AGC (demod)
- Ring-buffer audio playback (audio_output)
- ML anomaly detection pipeline (ml/)
- PyQt5 GUI with spectrum, waterfall, map, and sensor tabs (gui_main)
"""



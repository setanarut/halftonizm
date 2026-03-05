"""Halftonizm dock presets.

Edit PRESETS to add your own entries.
Each preset can define any or all supported settings.


"""

PRESETS = {
    "Default": {
        "WAVE_COUNT": 12,
        "TOTAL_FRAMES": 8,
        "WAVEFORM": "sawtooth",
        "REVERSE": False,
        "FPS": 12,
        "BLENDING_MODE": "HardMix",
        "HARD_MIX": True,
        "RESULT_SCALE": "%50",
    },
    "Smoothstep": {
        "WAVE_COUNT": 12,
        "TOTAL_FRAMES": 8,
        "WAVEFORM": "sawtooth",
        "REVERSE": False,
        "FPS": 12,
        "BLENDING_MODE": "Smoothstep",
        "HARD_MIX": True,
        "RESULT_SCALE": "%50",
    },
    "Flow Map Waves": {
        "WAVE_COUNT": 16,
        "TOTAL_FRAMES": 16,
        "WAVEFORM": "sawtooth",
        "REVERSE": False,
        "FPS": 12,
        "BLENDING_MODE": "Binary",
        "HARD_MIX": False,
        "RESULT_SCALE": "%50",
    },
}

# BBD's Krita Script Starter Feb 2018
import array
import contextlib
import importlib
import io
import math
import multiprocessing
import os
import ssl
import site
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed

try:
    from . import presets as presets_module
except Exception:
    try:
        import presets as presets_module
    except Exception:
        presets_module = None
from krita import DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QImage, QMovie, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

DOCKER_NAME = "Halftonizm"
DOCKER_ID = "pykrita_halftonizm"

ARTWORK_LAYERS_GROUP_NAME = "Artwork layers"
FLOW_MAP_LAYER_NAME = "Flow Map"


def _ensure_user_site_on_path():
    try:
        user_sites = site.getusersitepackages()
    except Exception:
        return
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    for path in user_sites:
        if path and path not in sys.path:
            sys.path.append(path)


_ensure_user_site_on_path()
np = None
_numpy_import_error = None
_NUMPY_DTYPE_BY_FMT = {
    "B": "uint8",
    "H": "uint16",
    "e": "float16",
    "f": "float32",
}


def _load_numpy():
    global np, _numpy_import_error
    if np is not None:
        return np
    try:
        import numpy as _np
    except Exception as err:
        np = None
        _numpy_import_error = str(err)
        return None
    np = _np
    _numpy_import_error = None
    return np


pil_image = None
_pillow_import_error = None


def _load_pillow_image():
    global pil_image, _pillow_import_error
    if pil_image is not None:
        return pil_image
    try:
        from PIL import Image as _Image
    except Exception as err:
        pil_image = None
        _pillow_import_error = str(err)
        return None
    pil_image = _Image
    _pillow_import_error = None
    return pil_image


# ---------------------------------------------------------------------------
# Module-level worker helpers for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------
_worker_state: dict = {}
_fork_worker_init_blob = None


def _worker_init(
    dem_bytes,
    base_bytes,
    overlay_bytes_list,
    overlay_flat_colors,
    use_artwork_mix,
    fmt_str,
    w,
    h,
    ch_count,
    flow_channel_idx,
    alpha_idx,
    max_val,
    wave_spacing,
    waveform,
    hard_mix_mode,
):
    """Runs once per worker process; caches pixel data so it isn't re-sent per frame."""
    global _fork_worker_init_blob
    _worker_state.clear()
    if dem_bytes is None and _fork_worker_init_blob is not None:
        dem_bytes, base_bytes, overlay_bytes_list, overlay_flat_colors = (
            _fork_worker_init_blob
        )
    _worker_state["numpy_enabled"] = False
    _worker_state.update(
        dict(
            w=w,
            h=h,
            ch_count=ch_count,
            flow_channel_idx=flow_channel_idx,
            alpha_idx=alpha_idx,
            use_artwork_mix=use_artwork_mix,
            max_val=max_val,
            wave_spacing=wave_spacing,
            waveform=waveform,
            hard_mix_mode=hard_mix_mode,
        )
    )

    np_mod = _load_numpy()
    if np_mod is not None and fmt_str in _NUMPY_DTYPE_BY_FMT:
        try:
            dtype = np_mod.dtype(_NUMPY_DTYPE_BY_FMT[fmt_str])
            dem_arr = np_mod.frombuffer(dem_bytes, dtype=dtype).reshape(
                (h, w, ch_count)
            )
            dem_phase = dem_arr[..., flow_channel_idx].astype(np_mod.float32) / float(
                wave_spacing
            )

            base_rgb = None
            if base_bytes is not None:
                base_arr = np_mod.frombuffer(base_bytes, dtype=dtype).reshape(
                    (h, w, ch_count)
                )
                base_rgb = base_arr[..., (2, 1, 0)].astype(np_mod.float32) / float(
                    max_val
                )

            overlay_alpha_list = []
            overlay_src_list = []
            normalized_flat_colors = list(overlay_flat_colors or [])
            for layer_idx, layer_bytes in enumerate(overlay_bytes_list or []):
                layer_arr = np_mod.frombuffer(layer_bytes, dtype=dtype).reshape(
                    (h, w, ch_count)
                )
                alpha_norm = layer_arr[..., alpha_idx].astype(np_mod.float32) / float(
                    max_val
                )
                overlay_alpha_list.append(alpha_norm)
                if layer_idx < len(normalized_flat_colors):
                    overlay_src_list.append(
                        np_mod.array(
                            normalized_flat_colors[layer_idx], dtype=np_mod.float32
                        )
                    )
                else:
                    rgb_norm = layer_arr[..., (2, 1, 0)].astype(np_mod.float32) / float(
                        max_val
                    )
                    safe_alpha = np_mod.where(alpha_norm > 1e-8, alpha_norm, 1.0)
                    overlay_src = np_mod.where(
                        alpha_norm[..., None] > 1e-8,
                        np_mod.clip(rgb_norm / safe_alpha[..., None], 0.0, 1.0),
                        0.0,
                    )
                    overlay_src_list.append(
                        overlay_src.astype(np_mod.float32, copy=False)
                    )

            _worker_state["np_mod"] = np_mod
            _worker_state["dem_phase"] = dem_phase
            _worker_state["base_rgb"] = base_rgb
            _worker_state["overlay_alpha_list"] = overlay_alpha_list
            _worker_state["overlay_src_list"] = overlay_src_list
            _worker_state["numpy_enabled"] = True
            return
        except Exception as err:
            _worker_state["numpy_init_error"] = str(err)

    _worker_state["vals_dem"] = array.array(fmt_str)
    _worker_state["vals_dem"].frombytes(dem_bytes)
    _worker_state["vals_base"] = None
    if base_bytes is not None:
        _worker_state["vals_base"] = array.array(fmt_str)
        _worker_state["vals_base"].frombytes(base_bytes)
    vals_overlays = []
    for layer_bytes in overlay_bytes_list or []:
        vals = array.array(fmt_str)
        vals.frombytes(layer_bytes)
        vals_overlays.append(vals)
    _worker_state["vals_overlays"] = vals_overlays
    _worker_state["overlay_flat_colors"] = list(overlay_flat_colors or [])


def _worker_compute_frame_numpy(t, s):
    np_mod = s["np_mod"]
    dem_phase = s["dem_phase"]
    use_artwork_mix = s["use_artwork_mix"]
    waveform = s["waveform"]
    hard_mix_mode = s["hard_mix_mode"]

    frac = np_mod.mod(dem_phase + np_mod.float32(t), 1.0)
    if waveform == "triangle":
        wv = 1.0 - np_mod.abs(2.0 * frac - 1.0)
    elif waveform == "sine":
        wv = 0.5 + 0.5 * np_mod.sin(frac * (2.0 * np_mod.pi))
    else:
        wv = frac

    if not use_artwork_mix:
        gray = np_mod.clip(
            np_mod.rint(np_mod.clip(wv, 0.0, 1.0) * 255.0), 0, 255
        ).astype(np_mod.uint8)
        rgba = np_mod.empty((s["h"], s["w"], 4), dtype=np_mod.uint8)
        rgba[..., 0:3] = gray[..., None]
        rgba[..., 3] = 255
        return rgba.tobytes()

    base_rgb = s["base_rgb"]
    if base_rgb is None:
        out_rgb = np_mod.zeros((s["h"], s["w"], 3), dtype=np_mod.float32)
    else:
        out_rgb = base_rgb.copy()

    for layer_idx, alpha_norm in enumerate(s["overlay_alpha_list"]):
        if hard_mix_mode == "Binary":
            mix = (alpha_norm - wv >= 0.0).astype(np_mod.float32)
            if not mix.any():
                continue
        else:
            score = alpha_norm - wv
            t2 = np_mod.clip((score + 0.18) / 0.36, 0.0, 1.0)
            mix = t2 * t2 * (3.0 - 2.0 * t2)
            if not np_mod.any(mix > 0.0):
                continue
        inv = 1.0 - mix
        src_rgb = s["overlay_src_list"][layer_idx]
        out_rgb *= inv[..., None]
        out_rgb += src_rgb * mix[..., None]

    rgb8 = np_mod.clip(
        np_mod.rint(np_mod.clip(out_rgb, 0.0, 1.0) * 255.0), 0, 255
    ).astype(np_mod.uint8)
    rgba = np_mod.empty((s["h"], s["w"], 4), dtype=np_mod.uint8)
    rgba[..., 0:3] = rgb8
    rgba[..., 3] = 255
    return rgba.tobytes()


def _worker_compute_frame(t):
    """Compute a single result frame (RGBA bytes) given animation offset t."""
    s = _worker_state
    if s.get("numpy_enabled"):
        return _worker_compute_frame_numpy(t, s)

    w = s["w"]
    h = s["h"]
    ch_count = s["ch_count"]
    flow_channel_idx = s["flow_channel_idx"]
    alpha_idx = s["alpha_idx"]
    use_artwork_mix = s["use_artwork_mix"]
    max_val = s["max_val"]
    wave_spacing = s["wave_spacing"]
    waveform = s["waveform"]
    hard_mix_mode = s["hard_mix_mode"]
    vals_dem = s["vals_dem"]
    vals_base = s["vals_base"]
    vals_overlays = s["vals_overlays"]
    overlay_flat_colors = s["overlay_flat_colors"]

    def fract(x):
        return x - math.floor(x)

    def wave(frac):
        if waveform == "triangle":
            return 1.0 - abs(2.0 * frac - 1.0)
        if waveform == "sawtooth":
            return frac
        if waveform == "sine":
            return 0.5 + 0.5 * math.sin(frac * 2 * math.pi)
        return frac

    def clamp01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def alpha_threshold_mask(wv, alpha_norm):
        # Alpha controls duty cycle/line thickness:
        # low alpha => thinner white, high alpha => thicker white.
        score = alpha_norm - wv
        if hard_mix_mode == "Binary":
            return 1.0 if score >= 0.0 else 0.0
        edge = 0.18
        t2 = clamp01((score + edge) / (2.0 * edge))
        return t2 * t2 * (3.0 - 2.0 * t2)

    def pixel_rgb_norm(vals, offset, unpremultiply=False):
        b = vals[offset + 0] / max_val
        g = vals[offset + 1] / max_val
        r = vals[offset + 2] / max_val
        if not unpremultiply:
            return (r, g, b)
        a = vals[offset + alpha_idx] / max_val
        if a <= 1e-8:
            return (0.0, 0.0, 0.0)
        inv_a = 1.0 / a
        return (clamp01(r * inv_a), clamp01(g * inv_a), clamp01(b * inv_a))

    result_pixels = bytearray(w * h * 4)
    for i in range(w * h):
        src_off = i * ch_count
        dst_off = i * 4
        v_dem = vals_dem[src_off + flow_channel_idx]
        frac = fract(v_dem / wave_spacing + t)
        wv = wave(frac)
        if not use_artwork_mix:
            wave_byte = int(round(clamp01(wv) * 255.0))
            result_pixels[dst_off + 0] = wave_byte
            result_pixels[dst_off + 1] = wave_byte
            result_pixels[dst_off + 2] = wave_byte
            result_pixels[dst_off + 3] = 255
            continue
        if vals_base is None:
            out_r, out_g, out_b = (0.0, 0.0, 0.0)
        else:
            out_r, out_g, out_b = pixel_rgb_norm(vals_base, src_off)
        for layer_idx, layer_vals in enumerate(vals_overlays):
            alpha_norm = layer_vals[src_off + alpha_idx] / max_val
            mix = alpha_threshold_mask(wv, alpha_norm)
            if mix <= 0.0:
                continue
            if layer_idx < len(overlay_flat_colors):
                src_r, src_g, src_b = overlay_flat_colors[layer_idx]
            else:
                src_r, src_g, src_b = pixel_rgb_norm(
                    layer_vals, src_off, unpremultiply=True
                )
            inv = 1.0 - mix
            out_r = src_r * mix + out_r * inv
            out_g = src_g * mix + out_g * inv
            out_b = src_b * mix + out_b * inv
        result_pixels[dst_off + 0] = int(round(clamp01(out_r) * 255.0))
        result_pixels[dst_off + 1] = int(round(clamp01(out_g) * 255.0))
        result_pixels[dst_off + 2] = int(round(clamp01(out_b) * 255.0))
        result_pixels[dst_off + 3] = 255

    return bytes(result_pixels)


DEPTH_META = {
    "U8": ("B", 255, 1),
    "U16": ("H", 65535, 2),
    "F16": ("e", 1.0, 2),
    "F32": ("f", 1.0, 4),
}


class UserCancelledError(RuntimeError):
    pass


class LayerRefreshComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_popup_callback = None

    def set_before_popup_callback(self, callback):
        self._before_popup_callback = callback

    def showPopup(self):
        if callable(self._before_popup_callback):
            try:
                self._before_popup_callback()
            except Exception:
                pass
        super().showPopup()


class Halftonizm(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(DOCKER_NAME)
        self._result_frames = []
        self._result_pixmaps = []
        self._result_fps = 12
        self._result_movie = None
        self._result_movie_path = None
        self._playback_timer = None
        self._playback_index = 0
        self._preview_source_pixmap = None
        self._preset_entries = {}
        self._logo_label = None
        self._logo_source_pixmap = None
        self._build_ui()
        self._sync_result_ratio_from_active_doc()
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_application_focus_changed)

    def _build_ui(self):
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel(container)
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        logo_label.setScaledContents(False)
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.isfile(logo_path):
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                self._logo_label = logo_label
                self._logo_source_pixmap = logo_pixmap
                logo_label.setMinimumHeight(logo_pixmap.height())
                self._update_logo_pixmap()
                logo_label.setContentsMargins(0, 0, 0, 6)
                layout.addWidget(logo_label)

        controls_grid = QGridLayout()
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setHorizontalSpacing(8)
        controls_grid.setVerticalSpacing(8)

        self.wave_count_spin = QSpinBox(container)
        self.wave_count_spin.setRange(1, 1024)
        self.wave_count_spin.setValue(8)

        self.total_frames_spin = QSpinBox(container)
        self.total_frames_spin.setRange(1, 9999)
        self.total_frames_spin.setValue(12)

        self.waveform_combo = QComboBox(container)
        self.waveform_combo.addItems(["triangle", "sawtooth", "sine"])
        self.waveform_combo.setCurrentText("sawtooth")

        self.reverse_check = QCheckBox(container)
        self.reverse_check.setChecked(False)

        self.fps_spin = QSpinBox(container)
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(12)

        self.blending_mode_combo = QComboBox(container)
        self.blending_mode_combo.addItems(["HardMix", "Smoothstep"])
        self.blending_mode_combo.setCurrentText("Smoothstep")
        self.hard_mix_check = QCheckBox(container)
        self.hard_mix_check.setChecked(True)

        self.result_scale_combo = QComboBox(container)
        self.result_scale_combo.addItems(["%100", "%72", "%50", "%25"])
        self.result_scale_combo.setCurrentText("%100")
        self.result_scale_combo.currentTextChanged.connect(
            self._on_result_scale_changed
        )
        self.output_size_value_label = QLabel("-", container)
        self.output_size_value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.flow_map_layer_combo = LayerRefreshComboBox(container)
        self.input_image_layer_combo = LayerRefreshComboBox(container)
        self.flow_map_layer_combo.set_before_popup_callback(
            self._refresh_layer_dropdowns
        )
        self.input_image_layer_combo.set_before_popup_callback(
            self._refresh_layer_dropdowns
        )

        self.settings_preset_combo = QComboBox(container)
        self.settings_preset_combo.addItem("Default")

        self.build_frames_button = QPushButton("Build Frames", container)
        self.build_frames_button.clicked.connect(self.result)
        self.preview_first_frame_button = QPushButton("Preview First Frame", container)
        self.preview_first_frame_button.clicked.connect(self.preview_first_frame)
        self.play_stop_button = QPushButton("Play", container)
        self.play_stop_button.setEnabled(False)
        self.play_stop_button.clicked.connect(self._toggle_result_playback)
        self.save_gif_button = QPushButton("Save GIF", container)
        self.save_gif_button.setEnabled(False)
        self.save_gif_button.clicked.connect(self.save_gif)
        self.save_apng_button = QPushButton("Save APNG", container)
        self.save_apng_button.setEnabled(False)
        self.save_apng_button.clicked.connect(self.save_apng)
        self.save_sequence_button = QPushButton("Save Image Seq", container)
        self.save_sequence_button.setEnabled(False)
        self.save_sequence_button.clicked.connect(self.save_image_sequence)
        self.numpy_install_button = QPushButton("Install NumPy", container)
        self.numpy_install_button.clicked.connect(self._install_numpy)
        self.numpy_status_label = QLabel("", container)
        self.numpy_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pillow_install_button = QPushButton("Install Pillow", container)
        self.pillow_install_button.clicked.connect(self._install_pillow)
        self.pillow_status_label = QLabel("", container)
        self.pillow_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        def make_control_cell(title, widget):
            cell = QWidget(container)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            title_label = QLabel(title, cell)
            cell_layout.addWidget(title_label)
            cell_layout.addWidget(widget)
            return cell

        control_cells = [
            ("Preset", self.settings_preset_combo),
            ("Waveform", self.waveform_combo),
            ("Blending mode", self.blending_mode_combo),
            ("Flow map Layer", self.flow_map_layer_combo),
            ("Artwork layers", self.input_image_layer_combo),
            ("Scale", self.result_scale_combo),
            ("Wave count", self.wave_count_spin),
            ("Total frames", self.total_frames_spin),
            ("FPS", self.fps_spin),
            ("Reverse", self.reverse_check),
            ("HardMix", self.hard_mix_check),
            ("Output size", self.output_size_value_label),
        ]
        for idx, (title, widget) in enumerate(control_cells):
            row = idx // 3
            col = idx % 3
            controls_grid.addWidget(make_control_cell(title, widget), row, col)

        buttons_grid = QGridLayout()
        buttons_grid.setContentsMargins(0, 0, 0, 0)
        buttons_grid.setHorizontalSpacing(8)
        buttons_grid.setVerticalSpacing(8)
        buttons_grid.addWidget(self.build_frames_button, 0, 0)
        buttons_grid.addWidget(self.preview_first_frame_button, 0, 1)
        buttons_grid.addWidget(self.save_gif_button, 1, 0)
        buttons_grid.addWidget(self.save_apng_button, 1, 1)
        buttons_grid.addWidget(self.save_sequence_button, 1, 2)

        numpy_grid = QGridLayout()
        numpy_grid.setContentsMargins(0, 0, 0, 0)
        numpy_grid.setHorizontalSpacing(8)
        numpy_grid.setVerticalSpacing(4)
        numpy_grid.addWidget(self.numpy_install_button, 0, 0)
        numpy_grid.addWidget(self.numpy_status_label, 0, 1, 1, 2)
        numpy_grid.addWidget(self.pillow_install_button, 1, 0)
        numpy_grid.addWidget(self.pillow_status_label, 1, 1, 1, 2)

        self.result_image_label = QLabel("Result frames will appear here.", container)
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumHeight(140)
        self.result_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.result_image_label.setScaledContents(False)
        self.result_image_label.setStyleSheet(
            "QLabel { background-color: #202020; color: #b8b8b8; border: 1px solid #3a3a3a; }"
        )

        layout.addLayout(controls_grid)
        layout.addLayout(buttons_grid)
        layout.addLayout(numpy_grid)
        layout.addWidget(self.result_image_label, 1)
        layout.addWidget(self.play_stop_button)
        self.setWidget(container)
        self.setMinimumHeight(600)
        self._reload_setting_presets()
        self.settings_preset_combo.currentTextChanged.connect(
            self._on_settings_preset_changed
        )
        self._update_numpy_status_label()
        self._update_pillow_status_label()
        self._refresh_layer_dropdowns()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_logo_pixmap()

    def _update_logo_pixmap(self):
        if self._logo_label is None or self._logo_source_pixmap is None:
            return
        available_w = self._logo_label.width()
        if available_w <= 1:
            if self.widget() is not None:
                available_w = self.widget().width()
        if available_w <= 1:
            available_w = self.width()
        if available_w <= 1:
            return
        min_w = self._logo_source_pixmap.width()
        target_w = max(min_w, available_w)
        scaled = self._logo_source_pixmap.scaledToWidth(
            max(1, target_w), Qt.SmoothTransformation
        )
        self._logo_label.setPixmap(scaled)

    def _normalize_preset_key(self, key):
        if key is None:
            return ""
        return "".join(ch for ch in str(key).upper() if ch.isalnum())

    def _to_bool(self, value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _normalize_scale_text(self, value):
        if isinstance(value, (int, float)):
            as_int = int(round(float(value)))
            return "%{}".format(as_int)
        text = str(value).strip()
        if not text:
            return None
        if text.startswith("%"):
            return text
        if text.endswith("%"):
            return "%{}".format(text[:-1].strip())
        return "%{}".format(text)

    def _trim_command_output(self, text, limit=3500):
        if text is None:
            return ""
        compact = str(text).strip()
        if len(compact) <= limit:
            return compact
        return "... (truncated) ...\n{}".format(compact[-limit:])

    def _run_pip_command(self, args):
        try:
            from pip._internal.cli.main import main as pip_main
        except Exception:
            try:
                from pip._internal import main as pip_main
            except Exception as err:
                return 1, "pip import failed: {}".format(err)

        output = io.StringIO()
        code = 1
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                result = pip_main(list(args))
                if isinstance(result, int):
                    code = result
                elif result is None:
                    code = 0
                else:
                    code = int(result)
            except SystemExit as exit_err:
                if isinstance(exit_err.code, int):
                    code = exit_err.code
                elif exit_err.code is None:
                    code = 0
                else:
                    code = 1
            except Exception as err:
                code = 1
                print("pip execution failed: {}".format(err))
        return code, output.getvalue()

    def _configure_pip_tls_environment(self):
        invalid_env = {}
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT"):
            value = os.environ.get(key)
            if value and not os.path.isfile(value):
                invalid_env[key] = value
                os.environ.pop(key, None)

        cert_path = None
        try:
            import certifi

            candidate = certifi.where()
            if candidate and os.path.isfile(candidate):
                cert_path = candidate
        except Exception:
            cert_path = None

        if cert_path is None:
            try:
                from pip._vendor import certifi as pip_certifi

                candidate = pip_certifi.where()
                if candidate and os.path.isfile(candidate):
                    cert_path = candidate
            except Exception:
                cert_path = None

        if cert_path is None:
            try:
                default_paths = ssl.get_default_verify_paths()
                for candidate in (
                    getattr(default_paths, "cafile", None),
                    getattr(default_paths, "openssl_cafile", None),
                ):
                    if candidate and os.path.isfile(candidate):
                        cert_path = candidate
                        break
            except Exception:
                cert_path = None

        if cert_path is None:
            for candidate in (
                "/etc/ssl/cert.pem",
                "/private/etc/ssl/cert.pem",
                "/etc/openssl/cert.pem",
            ):
                if os.path.isfile(candidate):
                    cert_path = candidate
                    break

        if cert_path is not None:
            os.environ["SSL_CERT_FILE"] = cert_path
            os.environ["REQUESTS_CA_BUNDLE"] = cert_path
            os.environ["PIP_CERT"] = cert_path

        return invalid_env, cert_path

    def _bootstrap_pip_with_get_pip(self):
        url = "https://bootstrap.pypa.io/get-pip.py"
        script_text = ""
        output = io.StringIO()
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                raw = resp.read()
            script_text = raw.decode("utf-8", errors="replace")
        except Exception as err:
            return 1, "get-pip download failed: {}".format(err)

        code = 1
        old_argv = list(sys.argv)
        old_sys_path = list(sys.path)
        old_env = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory(prefix="halftonizm_getpip_") as tmpdir:
                script_path = os.path.join(tmpdir, "get-pip.py")
                with open(script_path, "w", encoding="utf-8") as handle:
                    handle.write(script_text)

                # Execute get-pip in-process so it installs into Krita's Python env.
                globals_ns = {
                    "__name__": "__main__",
                    "__file__": script_path,
                    "__package__": None,
                    "__cached__": None,
                }
                sys.argv = [script_path, "--user"]
                if sys.path:
                    sys.path[0] = tmpdir
                else:
                    sys.path.append(tmpdir)
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(
                    output
                ):
                    try:
                        with open(script_path, "r", encoding="utf-8") as script_handle:
                            compiled = compile(
                                script_handle.read(), script_path, "exec"
                            )
                        exec(compiled, globals_ns, globals_ns)
                        code = 0
                    except SystemExit as exit_err:
                        if isinstance(exit_err.code, int):
                            code = exit_err.code
                        elif exit_err.code is None:
                            code = 0
                        else:
                            code = 1
        except Exception as err:
            print("get-pip execution failed: {}".format(err), file=output)
            code = 1
        finally:
            sys.argv = old_argv
            sys.path[:] = old_sys_path
            os.environ.clear()
            os.environ.update(old_env)

        return code, output.getvalue()

    def _clear_module_prefixes(self, prefixes):
        normalized = [p for p in prefixes if p]
        if not normalized:
            return
        for mod_name in list(sys.modules.keys()):
            for prefix in normalized:
                if mod_name == prefix or mod_name.startswith(prefix + "."):
                    sys.modules.pop(mod_name, None)
                    break

    def _ensure_pip_ready(self, logs):
        try:
            import pip  # noqa: F401

            logs.append("pip module already available in Krita Python.")
            return
        except Exception:
            pass

        pip_bootstrapped = False
        ensure_output = io.StringIO()
        try:
            import ensurepip

            with contextlib.redirect_stdout(ensure_output), contextlib.redirect_stderr(
                ensure_output
            ):
                ensurepip.bootstrap(upgrade=True, user=True)
            logs.append(
                "ensurepip bootstrap output:\n{}".format(
                    ensure_output.getvalue().strip()
                )
            )
            pip_bootstrapped = True
        except Exception as err:
            logs.append(
                "ensurepip unavailable/failed: {}\n{}".format(
                    err,
                    self._trim_command_output(ensure_output.getvalue()),
                )
            )
        if not pip_bootstrapped:
            code, output = self._bootstrap_pip_with_get_pip()
            logs.append("get-pip bootstrap output:\n{}".format(output.strip()))
            if code != 0:
                raise RuntimeError(
                    "pip bootstrap failed (ensurepip missing and get-pip failed)."
                )

    def _install_package_via_pip(self, package_name, logs):
        self._ensure_pip_ready(logs)
        self._clear_module_prefixes(["certifi", "pip._vendor.certifi"])

        invalid_env, cert_path = self._configure_pip_tls_environment()
        if invalid_env:
            logs.append(
                "Cleared invalid TLS env vars: {}".format(
                    ", ".join(
                        "{}={}".format(k, v) for k, v in sorted(invalid_env.items())
                    )
                )
            )
        if cert_path:
            logs.append("Using TLS CA bundle: {}".format(cert_path))
        else:
            logs.append("TLS CA bundle auto-detection failed; using default SSL paths.")

        install_attempts = [
            ["install", "--disable-pip-version-check", "--upgrade", package_name],
            [
                "install",
                "--disable-pip-version-check",
                "--user",
                "--upgrade",
                package_name,
            ],
            [
                "install",
                "--disable-pip-version-check",
                "--trusted-host",
                "pypi.org",
                "--trusted-host",
                "files.pythonhosted.org",
                "--user",
                "--upgrade",
                package_name,
            ],
        ]
        install_ok = False
        last_output = ""
        for args in install_attempts:
            code, output = self._run_pip_command(args)
            logs.append("pip {}\n{}".format(" ".join(args), output.strip()))
            last_output = output
            if code == 0:
                install_ok = True
                break
        if not install_ok:
            raise RuntimeError(
                "{} install command failed.\n{}".format(
                    package_name, self._trim_command_output(last_output)
                )
            )

    def _update_numpy_status_label(self):
        np_mod = _load_numpy()
        if np_mod is None:
            self.numpy_status_label.setText("NumPy: Not installed")
            self.numpy_status_label.setStyleSheet("QLabel { color: #c46363; }")
            self.numpy_install_button.setText("Install NumPy")
        else:
            self.numpy_status_label.setText(
                "NumPy: Installed ({})".format(getattr(np_mod, "__version__", "?"))
            )
            self.numpy_status_label.setStyleSheet("QLabel { color: #72c276; }")
            self.numpy_install_button.setText("Reinstall NumPy")

    def _update_pillow_status_label(self):
        pil_mod = _load_pillow_image()
        if pil_mod is None:
            self.pillow_status_label.setText("Pillow: Not installed")
            self.pillow_status_label.setStyleSheet("QLabel { color: #c46363; }")
            self.pillow_install_button.setText("Install Pillow")
        else:
            self.pillow_status_label.setText(
                "Pillow: Installed ({})".format(getattr(pil_mod, "__version__", "?"))
            )
            self.pillow_status_label.setStyleSheet("QLabel { color: #72c276; }")
            self.pillow_install_button.setText("Reinstall Pillow")

    def _install_package_ui(
        self, package_name, button, status_update_fn, after_install_fn
    ):
        title_name = package_name.strip()
        progress = QProgressDialog(
            "Installing {} into Krita Python...".format(title_name), "", 0, 0, self
        )
        progress.setWindowTitle("Halftonizm")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        logs = []
        try:
            button.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress.show()
            QApplication.processEvents()

            self._install_package_via_pip(package_name, logs)
            _ensure_user_site_on_path()
            after_install_fn()
            status_update_fn()
            self._show_info_message("{} install complete.".format(title_name))
        except Exception as err:
            status_update_fn()
            log_text = self._trim_command_output("\n\n".join(logs))
            if log_text:
                self._show_error_message(
                    "{} installation failed: {}\n\n{}".format(title_name, err, log_text)
                )
            else:
                self._show_error_message(
                    "{} installation failed: {}".format(title_name, err)
                )
        finally:
            progress.reset()
            progress.close()
            progress.deleteLater()
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            button.setEnabled(True)

    def _install_numpy(self):
        def _after_numpy_install():
            global np
            np = None
            self._clear_module_prefixes(["numpy"])
            importlib.invalidate_caches()
            np_mod = _load_numpy()
            if np_mod is None:
                reason = _numpy_import_error or "unknown import error"
                raise RuntimeError(
                    "NumPy import failed after install: {}".format(reason)
                )

        self._install_package_ui(
            "numpy",
            self.numpy_install_button,
            self._update_numpy_status_label,
            _after_numpy_install,
        )

    def _install_pillow(self):
        def _after_pillow_install():
            global pil_image
            pil_image = None
            self._clear_module_prefixes(["PIL"])
            importlib.invalidate_caches()
            pil_mod = _load_pillow_image()
            if pil_mod is None:
                reason = _pillow_import_error or "unknown import error"
                raise RuntimeError(
                    "Pillow import failed after install: {}".format(reason)
                )

        self._install_package_ui(
            "Pillow",
            self.pillow_install_button,
            self._update_pillow_status_label,
            _after_pillow_install,
        )

    def _read_presets_data(self):
        if presets_module is None:
            return {}
        data = getattr(presets_module, "PRESETS", None)
        if data is None:
            data = getattr(presets_module, "presets", None)
        if data is None:
            return {}

        parsed = {}
        if isinstance(data, dict):
            for name, values in data.items():
                if isinstance(values, dict):
                    parsed[str(name)] = values
            return parsed

        if isinstance(data, (list, tuple)):
            for idx, entry in enumerate(data):
                if not isinstance(entry, dict):
                    continue
                name = (
                    entry.get("name")
                    or entry.get("title")
                    or "Preset {}".format(idx + 1)
                )
                values = entry.get("values")
                if values is None:
                    values = entry.get("settings")
                if values is None:
                    values = {
                        k: v
                        for k, v in entry.items()
                        if k not in ("name", "title", "values", "settings")
                    }
                if isinstance(values, dict):
                    parsed[str(name)] = values
        return parsed

    def _reload_setting_presets(self):
        global presets_module
        if presets_module is not None:
            try:
                presets_module = importlib.reload(presets_module)
            except Exception:
                pass
        self._preset_entries = self._read_presets_data()
        combo = self.settings_preset_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("Custom")
            for name in self._preset_entries.keys():
                combo.addItem(name)
            if combo.findText("Default") >= 0:
                combo.setCurrentText("Default")
            elif combo.count() > 1:
                combo.setCurrentIndex(1)
            else:
                combo.setCurrentText("Custom")
        finally:
            combo.blockSignals(False)
        self._on_settings_preset_changed(combo.currentText())

    def _preset_get_value(self, preset_values, aliases):
        normalized = {}
        for key, value in preset_values.items():
            normalized[self._normalize_preset_key(key)] = value
        for alias in aliases:
            normalized_alias = self._normalize_preset_key(alias)
            if normalized_alias in normalized:
                return normalized[normalized_alias]
        return None

    def _apply_setting_preset(self, preset_values):
        value = self._preset_get_value(preset_values, ["WAVE_COUNT", "WAVECOUNT"])
        if value is not None:
            self.wave_count_spin.setValue(int(value))

        value = self._preset_get_value(
            preset_values,
            ["TOTAL_FRAMES", "TOTALFRAMES", "FRAMECOUNT", "FRAMES"],
        )
        if value is not None:
            self.total_frames_spin.setValue(int(value))

        value = self._preset_get_value(preset_values, ["WAVEFORM"])
        if value is not None:
            self.waveform_combo.setCurrentText(str(value))

        value = self._preset_get_value(preset_values, ["REVERSE"])
        if value is not None:
            self.reverse_check.setChecked(self._to_bool(value))

        value = self._preset_get_value(preset_values, ["FPS"])
        if value is not None:
            self.fps_spin.setValue(int(value))

        value = self._preset_get_value(
            preset_values,
            ["BLENDING_MODE", "BLENDINGMODE", "MODE"],
        )
        if value is not None:
            mode_text = str(value).strip().lower()
            if mode_text in ("hardmix", "hard_mix", "binary"):
                self.blending_mode_combo.setCurrentText("HardMix")
            elif mode_text in ("smoothstep", "smooth"):
                self.blending_mode_combo.setCurrentText("Smoothstep")

        value = self._preset_get_value(
            preset_values,
            ["HARD_MIX", "HARDMIX", "USE_HARDMIX", "MIX_ARTWORK"],
        )
        if value is not None:
            self.hard_mix_check.setChecked(self._to_bool(value))

        value = self._preset_get_value(
            preset_values, ["RESULT_SCALE", "RESULTSCALE", "SCALE"]
        )
        if value is not None:
            scale_text = self._normalize_scale_text(value)
            if scale_text:
                if self.result_scale_combo.findText(scale_text) < 0:
                    self.result_scale_combo.addItem(scale_text)
                self.result_scale_combo.setCurrentText(scale_text)
        self._update_output_size_label()

    def _on_settings_preset_changed(self, preset_name):
        if preset_name == "Custom":
            return
        preset_values = self._preset_entries.get(preset_name)
        if not preset_values:
            return
        try:
            self._apply_setting_preset(preset_values)
        except Exception as err:
            self._show_error_message(
                "Preset '{}' could not be applied: {}".format(preset_name, err)
            )

    def _sync_result_ratio_from_active_doc(self):
        self._update_output_size_label()

    def _parse_scale_factor(self):
        text = self.result_scale_combo.currentText().strip()
        if not text:
            return None
        if text.startswith("%"):
            text = text[1:].strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            scale = float(text) / 100.0
        except ValueError:
            return None
        if scale <= 0:
            return None
        return scale

    def _get_output_size_from_doc(self, doc=None):
        app = Krita.instance()
        if doc is None:
            if app is None:
                return None
            doc = app.activeDocument()
        if doc is None:
            return None
        scale = self._parse_scale_factor()
        if scale is None:
            return None
        width = max(1, int(round(doc.width() * scale)))
        height = max(1, int(round(doc.height() * scale)))
        return (width, height)

    def _compute_layer_flat_rgb(self, layer_bytes, fmt, max_val, ch_count, alpha_idx):
        np_mod = _load_numpy()
        if np_mod is not None and fmt in _NUMPY_DTYPE_BY_FMT:
            try:
                dtype = np_mod.dtype(_NUMPY_DTYPE_BY_FMT[fmt])
                vals = np_mod.frombuffer(layer_bytes, dtype=dtype)
                vals = vals.reshape((-1, ch_count))
                alpha = vals[:, alpha_idx].astype(np_mod.float32) / float(max_val)
                mask = alpha > 0.0
                if not np_mod.any(mask):
                    return (1.0, 1.0, 1.0)
                rgb = vals[:, (2, 1, 0)].astype(np_mod.float32) / float(max_val)
                safe_alpha = np_mod.where(alpha > 1e-8, alpha, 1.0)
                rgb = np_mod.where(
                    alpha[:, None] > 1e-8,
                    np_mod.clip(rgb / safe_alpha[:, None], 0.0, 1.0),
                    0.0,
                )
                weighted = (rgb[mask] * alpha[mask, None]).sum(axis=0)
                total_alpha = float(alpha[mask].sum())
                if total_alpha <= 1e-8:
                    return (1.0, 1.0, 1.0)
                out = np_mod.clip(weighted / total_alpha, 0.0, 1.0)
                return (float(out[0]), float(out[1]), float(out[2]))
            except Exception:
                pass

        vals = array.array(fmt)
        vals.frombytes(layer_bytes)
        weighted_r = 0.0
        weighted_g = 0.0
        weighted_b = 0.0
        total_alpha = 0.0
        for px in range(0, len(vals), ch_count):
            alpha = vals[px + alpha_idx] / max_val
            if alpha <= 0.0:
                continue
            b = vals[px + 0] / max_val
            g = vals[px + 1] / max_val
            r = vals[px + 2] / max_val
            # Krita pixelData can be premultiplied. Recover flat color first.
            if alpha > 1e-8:
                inv_a = 1.0 / alpha
                r = self._clamp01(r * inv_a)
                g = self._clamp01(g * inv_a)
                b = self._clamp01(b * inv_a)
            weighted_r += r * alpha
            weighted_g += g * alpha
            weighted_b += b * alpha
            total_alpha += alpha
        if total_alpha <= 1e-8:
            return (1.0, 1.0, 1.0)
        inv_total = 1.0 / total_alpha
        return (
            self._clamp01(weighted_r * inv_total),
            self._clamp01(weighted_g * inv_total),
            self._clamp01(weighted_b * inv_total),
        )

    def _update_output_size_label(self):
        size = self._get_output_size_from_doc()
        if size is None:
            self.output_size_value_label.setText("-")
        else:
            self.output_size_value_label.setText("{} x {}".format(size[0], size[1]))

    def _on_result_scale_changed(self, _value):
        self._update_output_size_label()

    def _delete_temp_movie_file(self):
        if self._result_movie_path and os.path.isfile(self._result_movie_path):
            try:
                os.remove(self._result_movie_path)
            except OSError:
                pass
        self._result_movie_path = None

    def _clear_result_movie(self):
        if self._playback_timer is not None:
            self._playback_timer.stop()
        if self._result_movie is not None:
            self._result_movie.stop()
            self.result_image_label.setMovie(None)
            self._result_movie.deleteLater()
            self._result_movie = None
        self._preview_source_pixmap = None
        self._delete_temp_movie_file()
        self.result_image_label.clear()

    def _is_widget_inside_docker(self, widget):
        node = widget
        while node is not None:
            if node is self:
                return True
            node = node.parentWidget()
        return False

    def _is_preview_playing(self):
        has_running_movie = (
            self._result_movie is not None
            and self._result_movie.state() == QMovie.Running
        )
        has_running_timer = (
            self._playback_timer is not None and self._playback_timer.isActive()
        )
        return has_running_movie or has_running_timer

    def _on_application_focus_changed(self, _old_widget, new_widget):
        if not self._is_preview_playing():
            return
        if new_widget is not None and self._is_widget_inside_docker(new_widget):
            return
        self._set_result_playing(False)

    def _set_preview_source_pixmap(self, pixmap):
        self._preview_source_pixmap = pixmap
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self):
        if self._preview_source_pixmap is None or self._preview_source_pixmap.isNull():
            return
        target_size = self.result_image_label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            self.result_image_label.setPixmap(self._preview_source_pixmap)
            return
        source_size = self._preview_source_pixmap.size()
        if source_size.width() <= 0 or source_size.height() <= 0:
            return

        # Fit to dock preview area, preserve aspect ratio, never upscale.
        scale_x = float(target_size.width()) / float(source_size.width())
        scale_y = float(target_size.height()) / float(source_size.height())
        scale = min(1.0, scale_x, scale_y)
        if scale < 1.0:
            fitted_size = QSize(
                max(1, int(round(source_size.width() * scale))),
                max(1, int(round(source_size.height() * scale))),
            )
            shown = self._preview_source_pixmap.scaled(
                fitted_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            shown = self._preview_source_pixmap
        self.result_image_label.setPixmap(shown)

    def _sync_result_movie_scale(self):
        if self._result_movie is None:
            self._refresh_preview_pixmap()
            return
        target_size = self.result_image_label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        frame_size = self._result_movie.currentPixmap().size()
        if frame_size.width() <= 0 or frame_size.height() <= 0:
            frame_size = self._result_movie.frameRect().size()
        if frame_size.width() <= 0 or frame_size.height() <= 0:
            return

        # Keep aspect ratio but do not upscale small previews.
        scale_x = float(target_size.width()) / float(frame_size.width())
        scale_y = float(target_size.height()) / float(frame_size.height())
        scale = min(1.0, scale_x, scale_y)
        fitted_size = QSize(
            max(1, int(round(frame_size.width() * scale))),
            max(1, int(round(frame_size.height() * scale))),
        )
        self._result_movie.setScaledSize(fitted_size)

    def _prepare_result_movie(self, progress=None):
        if not self._result_frames:
            self._clear_result_movie()
            self.play_stop_button.setEnabled(False)
            return False

        self._clear_result_movie()
        fd, movie_path = tempfile.mkstemp(prefix="halftonizm_result_", suffix=".gif")
        os.close(fd)
        try:
            self._write_result_gif(movie_path, progress=progress)
        except Exception:
            if os.path.isfile(movie_path):
                try:
                    os.remove(movie_path)
                except OSError:
                    pass
            raise

        movie = QMovie(movie_path)
        if not movie.isValid():
            try:
                os.remove(movie_path)
            except OSError:
                pass
            raise RuntimeError("Could not load generated result GIF with QMovie.")
        movie.setCacheMode(QMovie.CacheAll)
        self._result_movie = movie
        self._result_movie_path = movie_path
        self.result_image_label.setMovie(self._result_movie)
        self._sync_result_movie_scale()
        self.play_stop_button.setEnabled(True)
        return True

    def _set_result_playing(self, playing):
        has_movie = self._result_movie is not None
        has_frames = bool(self._result_frames)
        if not has_movie and not has_frames:
            self.play_stop_button.setText("Play")
            self.play_stop_button.setEnabled(False)
            return

        if self._playback_timer is None:
            self._playback_timer = QTimer(self)
            self._playback_timer.timeout.connect(self._advance_frame_preview)

        if playing:
            if has_movie:
                if self._result_movie.state() == QMovie.NotRunning:
                    self._result_movie.start()
                else:
                    self._result_movie.setPaused(False)
            else:
                interval_ms = max(
                    1, int(round(1000.0 / float(max(1, self._result_fps))))
                )
                self._playback_timer.start(interval_ms)
            self.play_stop_button.setText("Stop")
            return
        if has_movie and self._result_movie.state() != QMovie.NotRunning:
            self._result_movie.setPaused(True)
        if self._playback_timer is not None:
            self._playback_timer.stop()
        self.play_stop_button.setText("Play")
        self.play_stop_button.setEnabled(True)

    def _toggle_result_playback(self):
        has_movie = self._result_movie is not None
        has_frames = bool(self._result_frames)
        if not has_movie and not has_frames:
            return
        if has_movie:
            is_running = self._result_movie.state() == QMovie.Running
        else:
            is_running = (
                self._playback_timer is not None and self._playback_timer.isActive()
            )
        self._set_result_playing(not is_running)

    def _advance_frame_preview(self):
        if not self._result_frames:
            if self._playback_timer is not None:
                self._playback_timer.stop()
            return
        self._playback_index = (self._playback_index + 1) % len(self._result_frames)
        if self._result_pixmaps and self._playback_index < len(self._result_pixmaps):
            self._set_preview_source_pixmap(self._result_pixmaps[self._playback_index])
        else:
            self._set_preview_source_pixmap(
                QPixmap.fromImage(self._result_frames[self._playback_index])
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_result_movie_scale()

    def closeEvent(self, event):
        self._clear_result_movie()
        super().closeEvent(event)

    def _show_error_message(self, text):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setText(text)
        msg.setWindowTitle("Krita Script Error")
        msg.exec_()

    def _show_info_message(self, text):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle("Halftonizm")
        msg.exec_()

    def _node_path(self, node):
        names = []
        cur = node
        safety = 0
        while cur is not None and safety < 1024:
            name = ""
            try:
                name = cur.name() or ""
            except Exception:
                name = ""
            if name:
                names.append(name)
            try:
                cur = cur.parentNode()
            except Exception:
                cur = None
            safety += 1
        names.reverse()
        return "/".join(names)

    def _collect_leaf_layers(self, root_node):
        layers = []

        def walk(node):
            for child in node.childNodes():
                children = child.childNodes()
                if children:
                    walk(child)
                else:
                    if not self._is_selectable_leaf_node(child):
                        continue
                    layers.append((self._node_path(child), child))

        walk(root_node)
        return layers

    def _is_selectable_leaf_node(self, node):
        try:
            node_type = (node.type() or "").strip().lower()
        except Exception:
            node_type = ""
        return not node_type.endswith("mask")

    def _collect_group_layers(self, root_node):
        layers = []

        def walk(node):
            for child in node.childNodes():
                children = child.childNodes()
                if children:
                    layers.append((self._node_path(child), child))
                    walk(child)

        walk(root_node)
        return layers

    def _collect_visible_leaf_layers_bottom_to_top(self, root_node):
        layers = []

        def walk(node):
            # Krita childNodes() is already ordered from bottommost up.
            for child in node.childNodes():
                grandchildren = child.childNodes()
                if grandchildren:
                    walk(child)
                    continue
                if not self._is_selectable_leaf_node(child):
                    continue
                try:
                    if not child.visible():
                        continue
                except Exception:
                    pass
                layers.append(child)

        walk(root_node)
        return layers

    def _restore_layer_combo_selection(
        self, combo, layers, selected_path, default_layer_name
    ):
        if selected_path:
            idx = combo.findData(selected_path)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                return
        for idx, (_path, layer) in enumerate(layers):
            try:
                layer_name = layer.name() or ""
            except Exception:
                layer_name = ""
            if layer_name == default_layer_name:
                combo.setCurrentIndex(idx)
                return
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _refresh_layer_dropdowns(self, doc=None):
        app = Krita.instance()
        if doc is None and app is not None:
            doc = app.activeDocument()

        prev_flow_path = self.flow_map_layer_combo.currentData()
        prev_input_path = self.input_image_layer_combo.currentData()

        self.flow_map_layer_combo.blockSignals(True)
        self.input_image_layer_combo.blockSignals(True)
        try:
            self.flow_map_layer_combo.clear()
            self.input_image_layer_combo.clear()
            if doc is None:
                self.flow_map_layer_combo.addItem("(No active document)", None)
                self.input_image_layer_combo.addItem("(No active document)", None)
                self.flow_map_layer_combo.setEnabled(False)
                self.input_image_layer_combo.setEnabled(False)
                return

            flow_layers = self._collect_leaf_layers(doc.rootNode())
            group_layers = self._collect_group_layers(doc.rootNode())

            if not flow_layers:
                self.flow_map_layer_combo.addItem("(No layers found)", None)
                self.flow_map_layer_combo.setEnabled(False)
            else:
                for path, layer in flow_layers:
                    try:
                        label = layer.name() or path
                    except Exception:
                        label = path
                    self.flow_map_layer_combo.addItem(label, path)
                self.flow_map_layer_combo.setEnabled(True)
                self._restore_layer_combo_selection(
                    self.flow_map_layer_combo,
                    flow_layers,
                    prev_flow_path,
                    FLOW_MAP_LAYER_NAME,
                )

            if not group_layers:
                self.input_image_layer_combo.addItem("(No group layers found)", None)
                self.input_image_layer_combo.setEnabled(False)
            else:
                for path, layer in group_layers:
                    try:
                        label = layer.name() or path
                    except Exception:
                        label = path
                    self.input_image_layer_combo.addItem(label, path)
                self.input_image_layer_combo.setEnabled(True)
                self._restore_layer_combo_selection(
                    self.input_image_layer_combo,
                    group_layers,
                    prev_input_path,
                    ARTWORK_LAYERS_GROUP_NAME,
                )
        finally:
            self.flow_map_layer_combo.blockSignals(False)
            self.input_image_layer_combo.blockSignals(False)

    def _find_layer_by_path(self, doc, path):
        if not path:
            return None
        for node_path, node in self._collect_leaf_layers(doc.rootNode()):
            if node_path == path:
                return node
        return None

    def _find_group_layer_by_path(self, doc, path):
        if not path:
            return None
        for node_path, node in self._collect_group_layers(doc.rootNode()):
            if node_path == path:
                return node
        return None

    def canvasChanged(self, canvas):
        self._sync_result_ratio_from_active_doc()
        self._refresh_layer_dropdowns()

    def _clear_animation_cache(self, app):
        action = app.action("clear_animation_cache")
        if action is not None:
            action.trigger()

    def _clamp01(self, x):
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _cast_channel_value(self, value, depth):
        if depth in ("F16", "F32"):
            return float(value)
        return int(round(value))

    def _hard_mix_value(self, wv, v_norm, max_val, depth, mode):
        # score=0 is the original hard threshold.
        score = (wv + v_norm) - 1.0
        if mode == "Binary":
            norm_val = 1.0 if score >= 0.0 else 0.0
        else:
            # Smoothstep around threshold for softer antialias-like transition.
            edge = 0.18
            t = self._clamp01((score + edge) / (2.0 * edge))
            norm_val = t * t * (3.0 - 2.0 * t)
        return self._cast_channel_value(norm_val * max_val, depth)

    def _generate_result_frames(self, frame_count_override=None):
        progress = None
        cancelled = False
        forced_arrow_cursor = False
        try:
            self._set_result_playing(False)
            self.build_frames_button.setEnabled(False)
            self.preview_first_frame_button.setEnabled(False)
            self.play_stop_button.setEnabled(False)
            self.save_sequence_button.setEnabled(False)
            self.save_gif_button.setEnabled(False)
            self.save_apng_button.setEnabled(False)
            QApplication.setOverrideCursor(Qt.ArrowCursor)
            forced_arrow_cursor = True
            wave_count = self.wave_count_spin.value()
            total_frames = self.total_frames_spin.value()
            waveform = self.waveform_combo.currentText()
            reverse = self.reverse_check.isChecked()
            blending_mode = self.blending_mode_combo.currentText()
            use_artwork_mix = self.hard_mix_check.isChecked()
            hard_mix_mode = "Binary" if blending_mode == "HardMix" else "Smoothstep"
            fps = self.fps_spin.value()
            n = total_frames
            if frame_count_override is not None:
                n = max(1, int(frame_count_override))
            if _load_numpy() is None:
                self._update_numpy_status_label()
                self._show_error_message(
                    "NumPy is not installed in Krita Python. Click 'Install NumPy' first."
                )
                return
            self._update_numpy_status_label()

            progress = QProgressDialog(
                "Preparing result generation...", "Cancel", 0, n, self
            )
            progress.setWindowTitle("Halftonizm")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            QApplication.processEvents()

            app = Krita.instance()
            if app is None:
                self._show_error_message("Could not access Krita application instance.")
                return

            doc = app.activeDocument()
            if doc is None:
                self._show_error_message(
                    "No active document. Run Build Frames or Preview First Frame while a document is open."
                )
                return
            self._refresh_layer_dropdowns(doc=doc)

            w, h = doc.width(), doc.height()
            model = doc.colorModel()
            depth = doc.colorDepth()
            self._sync_result_ratio_from_active_doc()

            if model != "RGBA":
                self._show_error_message(
                    "Document color model must be RGBA. Current model: {}".format(model)
                )
                return
            if depth not in DEPTH_META:
                self._show_error_message("Unsupported colorDepth: {}".format(depth))
                return

            fmt, max_val, bytes_per_channel = DEPTH_META[depth]
            wave_spacing = max_val / wave_count
            ch_count = 4
            flow_channel_idx = 2
            alpha_idx = 3

            flow_map_path = self.flow_map_layer_combo.currentData()
            src_layer = self._find_layer_by_path(doc, flow_map_path)
            if src_layer is None:
                self._show_error_message(
                    "Selected flow map layer was not found. Please pick a valid layer."
                )
                return

            expected_len = w * h * ch_count
            expected_bytes = expected_len * bytes_per_channel
            progress.setLabelText(
                "Reading flow map layer: {}".format(
                    self.flow_map_layer_combo.currentText()
                )
            )
            QApplication.processEvents()
            dem_bytes = bytes(src_layer.pixelData(0, 0, w, h))
            if len(dem_bytes) != expected_bytes:
                self._show_error_message(
                    "Flow map pixel byte length differs from expected: {} != {} (model={}, depth={})".format(
                        len(dem_bytes), expected_bytes, model, depth
                    )
                )
                return

            base_bytes = None
            overlay_bytes_list = []
            overlay_flat_colors = []
            if use_artwork_mix:
                artwork_group_path = self.input_image_layer_combo.currentData()
                artwork_group = self._find_group_layer_by_path(doc, artwork_group_path)
                if artwork_group is None:
                    self._show_error_message(
                        "Selected artwork layers group was not found (Blending mode: {}).".format(
                            blending_mode
                        )
                    )
                    return

                stack_layers = self._collect_visible_leaf_layers_bottom_to_top(
                    artwork_group
                )
                if not stack_layers:
                    self._show_error_message(
                        "Selected artwork group has no visible paint layers."
                    )
                    return
                base_layer = stack_layers[0]
                overlay_layers = stack_layers[1:]
                progress.setLabelText(
                    "Reading artwork base layer: {}".format(
                        base_layer.name() or "(unnamed)"
                    )
                )
                QApplication.processEvents()
                base_bytes = bytes(base_layer.pixelData(0, 0, w, h))
                if len(base_bytes) != expected_bytes:
                    self._show_error_message(
                        "Base layer pixel byte length differs from expected: {} != {} (model={}, depth={})".format(
                            len(base_bytes), expected_bytes, model, depth
                        )
                    )
                    return
                for idx, layer in enumerate(overlay_layers):
                    progress.setLabelText(
                        "Reading artwork channel layer {}/{}: {}".format(
                            idx + 1, len(overlay_layers), layer.name() or "(unnamed)"
                        )
                    )
                    QApplication.processEvents()
                    layer_bytes = bytes(layer.pixelData(0, 0, w, h))
                    if len(layer_bytes) != expected_bytes:
                        self._show_error_message(
                            "Artwork channel layer pixel byte length differs from expected: {} != {} (model={}, depth={})".format(
                                len(layer_bytes), expected_bytes, model, depth
                            )
                        )
                        return
                    overlay_bytes_list.append(layer_bytes)
                    overlay_flat_colors.append(
                        self._compute_layer_flat_rgb(
                            layer_bytes, fmt, max_val, ch_count, alpha_idx
                        )
                    )

            step = 1.0 / n
            frames_t = [i * step for i in range(n)]
            if reverse:
                frames_t = list(reversed(frames_t))

            output_size = self._get_output_size_from_doc(doc=doc)
            if output_size is None:
                self._show_error_message(
                    "Invalid scale value. Please choose a valid Scale percentage."
                )
                return
            scaled_w, scaled_h = output_size
            self._update_output_size_label()
            pillow_mod = _load_pillow_image()
            if (scaled_w != w or scaled_h != h) and pillow_mod is None:
                self._update_pillow_status_label()
                self._show_error_message(
                    "Pillow is required for bicubic scaling. Click 'Install Pillow' first."
                )
                return
            self._update_pillow_status_label()

            progress.setLabelText("Preparing workers...")
            QApplication.processEvents()

            cancel_state = {"requested": False}

            def _on_cancel():
                cancel_state["requested"] = True
                progress.setLabelText("Cancelling...")
                QApplication.processEvents()

            progress.canceled.connect(_on_cancel)
            self._clear_result_movie()
            self._result_frames = []
            self._result_pixmaps = []
            self.play_stop_button.setText("Play")

            # Serialize pixel data once; workers load it via the initializer so
            # each worker process receives the arrays only once, not once per frame.
            # Use 'fork' so workers inherit already-loaded memory without
            # re-importing the module (which would fail on Krita's embedded Python).
            # Falls back to the default context on platforms that don't support fork.
            try:
                mp_ctx = multiprocessing.get_context("fork")
            except ValueError:
                mp_ctx = multiprocessing.get_context()

            use_fork_blob = False
            dem_init_bytes = dem_bytes
            base_init_bytes = base_bytes
            overlays_init_bytes = tuple(overlay_bytes_list)
            overlay_colors_init = tuple(overlay_flat_colors)
            try:
                start_method = mp_ctx.get_start_method()
            except Exception:
                start_method = None
            if start_method == "fork":
                global _fork_worker_init_blob
                _fork_worker_init_blob = (
                    dem_bytes,
                    base_bytes,
                    overlays_init_bytes,
                    overlay_colors_init,
                )
                dem_init_bytes = None
                base_init_bytes = None
                overlays_init_bytes = None
                overlay_colors_init = None
                use_fork_blob = True

            raw_results = [None] * n
            try:
                with ProcessPoolExecutor(
                    max_workers=os.cpu_count() or 4,
                    mp_context=mp_ctx,
                    initializer=_worker_init,
                    initargs=(
                        dem_init_bytes,
                        base_init_bytes,
                        overlays_init_bytes,
                        overlay_colors_init,
                        use_artwork_mix,
                        fmt,
                        w,
                        h,
                        ch_count,
                        flow_channel_idx,
                        alpha_idx,
                        max_val,
                        wave_spacing,
                        waveform,
                        hard_mix_mode,
                    ),
                ) as executor:
                    future_to_idx = {
                        executor.submit(_worker_compute_frame, t): idx
                        for idx, t in enumerate(frames_t)
                    }
                    done_count = 0
                    for future in as_completed(future_to_idx):
                        if cancel_state["requested"]:
                            cancelled = True
                            for f in future_to_idx:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        frame_idx = future_to_idx[future]
                        try:
                            raw_results[frame_idx] = future.result()
                        except CancelledError:
                            cancelled = True
                            break
                        done_count += 1
                        progress.setLabelText(
                            "Generating result frame {}/{}".format(done_count, n)
                        )
                        progress.setValue(done_count)
                        QApplication.processEvents()
            finally:
                if use_fork_blob:
                    _fork_worker_init_blob = None

            if cancelled:
                self._show_info_message("Result generation was cancelled by the user.")
                return
            result_frames = []
            progress.setLabelText("Scaling result frames...")
            QApplication.processEvents()
            for pixel_bytes in raw_results:
                if pixel_bytes is None:
                    continue
                if scaled_w != w or scaled_h != h:
                    resized_bytes = self._resize_rgba_bytes_with_pillow(
                        pixel_bytes, w, h, scaled_w, scaled_h
                    )
                    if resized_bytes is None:
                        self._show_error_message(
                            "Pillow bicubic scaling failed because Pillow is not available."
                        )
                        return
                    image = QImage(
                        resized_bytes,
                        scaled_w,
                        scaled_h,
                        scaled_w * 4,
                        QImage.Format_RGBA8888,
                    ).copy()
                else:
                    image = QImage(
                        pixel_bytes, w, h, w * 4, QImage.Format_RGBA8888
                    ).copy()
                result_frames.append(image)

            if not result_frames:
                self._show_error_message("No result frames were generated.")
                return

            self._result_frames = result_frames
            self._result_pixmaps = [QPixmap.fromImage(frame) for frame in result_frames]
            self._result_fps = fps
            self._playback_index = 0
            self.result_image_label.setText("")
            if self._result_pixmaps:
                self._set_preview_source_pixmap(self._result_pixmaps[0])
            else:
                self._set_preview_source_pixmap(QPixmap.fromImage(result_frames[0]))
            self.save_sequence_button.setEnabled(True)
            self.save_gif_button.setEnabled(True)
            self.save_apng_button.setEnabled(True)
            self._set_result_playing(False)

        except Exception as err:
            self._show_error_message("Error while generating result: {}".format(err))
        finally:
            if progress is not None:
                progress.reset()
                progress.close()
                progress.deleteLater()
            if forced_arrow_cursor and QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self.build_frames_button.setEnabled(True)
            self.preview_first_frame_button.setEnabled(True)
            has_preview = bool(self._result_frames) or self._result_movie is not None
            self.play_stop_button.setEnabled(has_preview)
            self.save_sequence_button.setEnabled(bool(self._result_frames))
            self.save_gif_button.setEnabled(bool(self._result_frames))
            self.save_apng_button.setEnabled(bool(self._result_frames))

    def preview_first_frame(self):
        self._generate_result_frames(frame_count_override=1)

    def result(self):
        self._generate_result_frames()

    def _pillow_bicubic_resample(self, image_mod):
        resampling = getattr(image_mod, "Resampling", None)
        if resampling is not None and hasattr(resampling, "BICUBIC"):
            return resampling.BICUBIC
        return image_mod.BICUBIC

    def _qimage_to_rgba_bytes(self, image):
        frame = image.convertToFormat(QImage.Format_RGBA8888)
        w = frame.width()
        h = frame.height()
        stride = frame.bytesPerLine()
        ptr = frame.constBits()
        ptr.setsize(stride * h)
        raw = bytes(ptr)
        if stride == w * 4:
            return w, h, raw

        np_mod = _load_numpy()
        if np_mod is not None:
            packed = (
                np_mod.frombuffer(raw, dtype=np_mod.uint8)
                .reshape((h, stride))[:, : w * 4]
                .copy()
                .tobytes()
            )
            return w, h, packed

        packed_bytes = bytearray(w * h * 4)
        for row in range(h):
            src_off = row * stride
            dst_off = row * w * 4
            packed_bytes[dst_off : dst_off + (w * 4)] = raw[src_off : src_off + (w * 4)]
        return w, h, bytes(packed_bytes)

    def _qimage_to_pillow_rgba(self, image):
        image_mod = _load_pillow_image()
        if image_mod is None:
            raise RuntimeError("Pillow is not installed.")
        w, h, rgba = self._qimage_to_rgba_bytes(image)
        return image_mod.frombytes("RGBA", (w, h), rgba)

    def _resize_rgba_bytes_with_pillow(self, pixel_bytes, src_w, src_h, dst_w, dst_h):
        image_mod = _load_pillow_image()
        if image_mod is None:
            return None
        src = image_mod.frombytes("RGBA", (src_w, src_h), pixel_bytes)
        resized = src.resize((dst_w, dst_h), self._pillow_bicubic_resample(image_mod))
        return resized.tobytes()

    def _write_result_gif(self, output_path, progress, frames=None, fps=None):
        source_frames = self._result_frames if frames is None else frames
        source_fps = self._result_fps if fps is None else fps
        if not source_frames:
            raise RuntimeError("No frames available for GIF encoding.")
        image_mod = _load_pillow_image()
        if image_mod is None:
            raise RuntimeError("Pillow is not installed. Install Pillow first.")

        duration_ms = max(1, int(round(1000.0 / max(1, source_fps))))
        total_frames = len(source_frames)
        pil_frames = []

        quantize_kwargs = {}
        quantize = getattr(image_mod, "Quantize", None)
        if quantize is not None and hasattr(quantize, "FASTOCTREE"):
            quantize_kwargs["method"] = quantize.FASTOCTREE
        elif hasattr(image_mod, "FASTOCTREE"):
            quantize_kwargs["method"] = image_mod.FASTOCTREE
        dither = getattr(image_mod, "Dither", None)
        if dither is not None and hasattr(dither, "NONE"):
            quantize_kwargs["dither"] = dither.NONE

        for idx, frame in enumerate(source_frames):
            if progress is not None and progress.wasCanceled():
                raise UserCancelledError("GIF export canceled by user.")
            pil_frame = self._qimage_to_pillow_rgba(frame).convert("RGB")
            pil_frames.append(pil_frame.quantize(colors=256, **quantize_kwargs))
            if progress is not None:
                progress.setLabelText(
                    "Encoding GIF frame {}/{}".format(idx + 1, total_frames)
                )
                progress.setValue(idx + 1)
                QApplication.processEvents()

        if not pil_frames:
            raise RuntimeError("No frames available for GIF encoding.")

        pil_frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )

    def _write_result_apng(self, output_path, progress, frames=None, fps=None):
        source_frames = self._result_frames if frames is None else frames
        source_fps = self._result_fps if fps is None else fps
        if not source_frames:
            raise RuntimeError("No frames available for APNG encoding.")
        image_mod = _load_pillow_image()
        if image_mod is None:
            raise RuntimeError("Pillow is not installed. Install Pillow first.")

        duration_ms = max(1, int(round(1000.0 / max(1, source_fps))))
        total_frames = len(source_frames)
        pil_frames = []
        for idx, frame in enumerate(source_frames):
            if progress is not None and progress.wasCanceled():
                raise UserCancelledError("APNG export canceled by user.")
            pil_frames.append(self._qimage_to_pillow_rgba(frame))
            if progress is not None:
                progress.setLabelText(
                    "Encoding APNG frame {}/{}".format(idx + 1, total_frames)
                )
                progress.setValue(idx + 1)
                QApplication.processEvents()

        if not pil_frames:
            raise RuntimeError("No frames available for APNG encoding.")

        pil_frames[0].save(
            output_path,
            format="PNG",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
            compress_level=3,
        )

    def save_image_sequence(self):
        if not self._result_frames:
            self._show_error_message(
                "No result frames to export. Generate Build Frames first."
            )
            return
        if _load_pillow_image() is None:
            self._update_pillow_status_label()
            self._show_error_message(
                "Pillow is required for fast PNG export. Click 'Install Pillow' first."
            )
            return

        start_dir = os.path.expanduser("~")
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Folder For Image Sequence", start_dir
        )
        if not output_dir:
            return

        total = len(self._result_frames)
        digits = max(4, len(str(total)))
        progress = QProgressDialog("Saving image sequence...", "Cancel", 0, total, self)
        progress.setWindowTitle("Halftonizm")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            for idx, frame in enumerate(self._result_frames):
                if progress.wasCanceled():
                    return
                name = "halftonizm_{:0{width}d}.png".format(idx + 1, width=digits)
                path = os.path.join(output_dir, name)
                pil_frame = self._qimage_to_pillow_rgba(frame)
                pil_frame.save(path, format="PNG", optimize=False, compress_level=3)
                progress.setLabelText("Saving frame {}/{}".format(idx + 1, total))
                progress.setValue(idx + 1)
                QApplication.processEvents()
        except Exception as err:
            self._show_error_message("Image sequence export failed: {}".format(err))
        finally:
            progress.close()
            progress.deleteLater()
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def save_gif(self):
        if not self._result_frames:
            self._show_error_message(
                "No result frames to export. Generate Build Frames first."
            )
            return
        if _load_pillow_image() is None:
            self._update_pillow_status_label()
            self._show_error_message(
                "Pillow is required for GIF export. Click 'Install Pillow' first."
            )
            return

        start_path = os.path.join(os.path.expanduser("~"), "halftonizm_result.gif")
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save GIF", start_path, "GIF Files (*.gif)"
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".gif"):
            output_path += ".gif"

        progress = QProgressDialog(
            "Encoding GIF frame 0/{}".format(len(self._result_frames)),
            "Cancel",
            0,
            len(self._result_frames),
            self,
        )
        progress.setWindowTitle("Halftonizm")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        try:
            progress.setLabelText("Preparing GIF export...")
            QApplication.processEvents()
            self._write_result_gif(output_path, progress=progress)
            progress.setLabelText("Finalizing GIF...")
            progress.setValue(len(self._result_frames))
            QApplication.processEvents()
        except UserCancelledError:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
        except Exception as err:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            self._show_error_message("GIF export failed: {}".format(err))
        finally:
            progress.reset()
            progress.close()
            progress.deleteLater()

    def save_apng(self):
        if not self._result_frames:
            self._show_error_message(
                "No result frames to export. Generate Build Frames first."
            )
            return
        if _load_pillow_image() is None:
            self._update_pillow_status_label()
            self._show_error_message(
                "Pillow is required for APNG export. Click 'Install Pillow' first."
            )
            return

        start_path = os.path.join(os.path.expanduser("~"), "halftonizm_result.png")
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save APNG", start_path, "PNG Files (*.png)"
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".png"):
            output_path += ".png"

        progress = QProgressDialog(
            "Encoding APNG frame 0/{}".format(len(self._result_frames)),
            "Cancel",
            0,
            len(self._result_frames),
            self,
        )
        progress.setWindowTitle("Halftonizm")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        try:
            progress.setLabelText("Preparing APNG export...")
            QApplication.processEvents()
            self._write_result_apng(output_path, progress=progress)
            progress.setLabelText("Finalizing APNG...")
            progress.setValue(len(self._result_frames))
            QApplication.processEvents()
        except UserCancelledError:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
        except Exception as err:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            self._show_error_message("APNG export failed: {}".format(err))
        finally:
            progress.reset()
            progress.close()
            progress.deleteLater()

    def run(self):
        # Backward-compatible alias for older bindings.
        self.result()


# Guard against re-import in worker processes (multiprocessing 'spawn'/'forkserver').
# Worker subprocesses only need the module-level helper functions above.
try:
    instance = Krita.instance()
    if instance is not None:
        dock_widget_factory = DockWidgetFactory(
            DOCKER_ID,
            DockWidgetFactoryBase.DockRight,
            Halftonizm,
        )
        instance.addDockWidgetFactory(dock_widget_factory)
except Exception:
    pass

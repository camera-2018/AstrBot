"""VAD Monitor API — tails AstrBot log for voice trace events and manages VAD config."""
import json
import os
import re
from pathlib import Path

from quart import jsonify, request

from astrbot.core import logger

from .route import Response, Route, RouteContext

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CONFIG_PATH = Path("data/napcat_voice_config.json")
_LOG_PATH = os.environ.get("ASTRBOT_LOG", "")


class VADMonitorRoute(Route):
    def __init__(self, context: RouteContext) -> None:
        super().__init__(context)
        self._log_pos = 0
        self._log_file = ""
        self.routes = [
            ("/vad/log", ("GET", self.vad_get_log)),
            ("/vad/config", ("GET", self.vad_get_config)),
            ("/vad/config", ("POST", self.vad_save_config)),
        ]
        self.register_routes()

    def _find_log_file(self) -> str:
        if _LOG_PATH:
            return _LOG_PATH
        # Check common log locations
        candidates = [
            Path("/tmp/astrbot.log"),  # nohup redirect
        ]
        # AstrBot data/logs/ directory
        log_dir = Path("data/logs")
        if log_dir.is_dir():
            logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(logs)
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                return str(c)
        return ""

    async def vad_get_log(self):
        try:
            offset = int(request.args.get("offset", self._log_pos))
        except (ValueError, TypeError):
            offset = self._log_pos

        log_file = self._log_file or self._find_log_file()
        if not log_file:
            return jsonify(Response().ok(data={"lines": [], "new_offset": 0}).__dict__)
        self._log_file = log_file

        try:
            size = os.path.getsize(log_file)
        except OSError:
            return jsonify(Response().ok(data={"lines": [], "new_offset": offset}).__dict__)

        if offset > size:
            offset = 0

        lines = []
        try:
            with open(log_file, "r", errors="replace") as f:
                f.seek(offset)
                for _ in range(500):
                    line = f.readline()
                    if not line:
                        break
                    if "napcat_voice trace=" in line:
                        lines.append(_ANSI_RE.sub("", line.rstrip()))
                new_offset = f.tell()
        except OSError:
            new_offset = offset

        self._log_pos = new_offset
        return jsonify(Response().ok(data={"lines": lines, "new_offset": new_offset}).__dict__)

    async def vad_get_config(self):
        try:
            cfg = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
        except Exception:
            cfg = {}
        return jsonify(Response().ok(data=cfg).__dict__)

    async def vad_save_config(self):
        try:
            data = await request.get_json()
            if not data or not isinstance(data, dict):
                return jsonify(Response().error("Invalid config data").__dict__)
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info("VAD config saved via WebUI: %s", data)
            return jsonify(Response().ok(message="Config saved").__dict__)
        except Exception as e:
            return jsonify(Response().error(str(e)).__dict__)

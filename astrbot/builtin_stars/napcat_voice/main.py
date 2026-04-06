import asyncio
import json
import os
import re
import shutil
import struct
import time
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import Any

from astrbot.api import llm_tool, logger, star
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.core.message.components import Plain
from astrbot.core.platform import AstrBotMessage, MessageMember
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.utils.active_event_registry import active_event_registry
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.media_utils import convert_audio_format


# ---------------------------------------------------------------------------
# Synthetic event — wraps STT text as a fake user message for AstrBot pipeline
# ---------------------------------------------------------------------------

class _SyntheticVoiceEvent(AiocqhttpMessageEvent):
    """Fake message event that captures the LLM reply text."""

    def __init__(
        self,
        *,
        source_event: AiocqhttpMessageEvent,
        recognized_text: str,
        injected_text: str | None = None,
        forward_to_chat: bool = True,
    ) -> None:
        effective_text = injected_text or recognized_text
        msg_obj = AstrBotMessage()
        msg_obj.type = source_event.get_message_type()
        msg_obj.self_id = source_event.get_self_id()
        msg_obj.session_id = source_event.session_id
        msg_obj.group_id = source_event.get_group_id()
        msg_obj.message_id = uuid.uuid4().hex
        msg_obj.sender = MessageMember(
            user_id=source_event.get_sender_id(),
            nickname=source_event.get_sender_name(),
        )
        msg_obj.message = [Plain(effective_text)]
        msg_obj.message_str = effective_text
        msg_obj.raw_message = effective_text
        msg_obj.timestamp = int(time.time())

        super().__init__(
            effective_text,
            msg_obj,
            source_event.platform_meta,
            source_event.session_id,
            source_event.bot,
        )
        self.session = source_event.session
        self.role = source_event.role
        self.is_wake = True
        self.is_at_or_wake_command = True
        self._extras.update(source_event.get_extra())
        self.set_extra("napcat_voice_origin", "call_stt")
        self._forward_to_chat = forward_to_chat
        self._reply_fragments: list[str] = []

    # -- reply capture --

    def _capture(self, message: MessageChain | None) -> None:
        if message is None:
            return
        plain = message.get_plain_text(with_other_comps_mark=False).strip()
        if plain:
            self._reply_fragments.append(plain)

    def get_reply_text(self) -> str:
        return "".join(f.strip() for f in self._reply_fragments if f.strip()).strip()

    async def wait_for_reply(self, timeout: float = 90.0) -> str:
        deadline = time.monotonic() + timeout
        seen = False
        last, stable = "", 0
        while time.monotonic() < deadline:
            is_active = active_event_registry.is_active(self)
            if is_active:
                seen = True
            cur = self.get_reply_text()
            if seen and not is_active:
                return cur
            if not is_active and cur:
                stable = stable + 1 if cur == last else 0
                if stable >= 4:
                    return cur
            last = cur
            await asyncio.sleep(0.2)
        return self.get_reply_text()

    async def send(self, message: MessageChain) -> None:
        self._capture(message)
        if self._forward_to_chat:
            await super().send(message)
        else:
            await AstrMessageEvent.send(self, message)

    async def send_streaming(self, generator, use_fallback: bool = False) -> None:
        async for chain in generator:
            await self.send(chain)


# ---------------------------------------------------------------------------
# OGG CRC-32 lookup table (polynomial 0x04C11DB7)
def _build_ogg_crc_table() -> list[int]:
    tbl: list[int] = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
        tbl.append(r)
    return tbl

_OGG_CRC_TABLE = _build_ogg_crc_table()


# Main plugin
# ---------------------------------------------------------------------------

class Main(star.Star):
    """NapCat Voice — VAD → STT → AstrBot → TTS voice call plugin."""

    _DEFAULT_VAD_CONFIG: dict[str, Any] = {
        "vad_start_threshold": 15,
        "vad_stop_threshold": 10,
        "vad_silence_frames": 15,
        "min_non_silent_frames": 3,
        "wait_seconds": 8.0,
        "poll_interval": 0.1,
        "max_frames": 150,
        "welcome_text": "我已接通，你可以直接说话。",
        "min_utterance_ms": 400,
        "post_speech_frames": 5,
        "pre_roll_size": 16,
        "adaptive_silence": True,
    }

    def __init__(self, context: star.Context) -> None:
        self.context = context
        self._active_call_user: dict[str, str] = {}
        self._auto_loop_tasks: dict[str, asyncio.Task[Any]] = {}
        self._auto_loop_resume_after_playback: dict[str, float] = {}
        self._auto_loop_drain_before_turn: dict[str, bool] = {}
        self._auto_loop_connected_context_pending: dict[str, bool] = {}
        self._tts_cache_dir = Path(get_astrbot_temp_path()) / "napcat_voice"
        self._tts_cache_dir.mkdir(parents=True, exist_ok=True)
        self._stt_lock = asyncio.Lock()
        self._call_locks: dict[str, asyncio.Lock] = {}
        self._vad_config = self._load_vad_config()

    # ── config ──

    def _load_vad_config(self) -> dict[str, Any]:
        config = dict(self._DEFAULT_VAD_CONFIG)
        path = Path("data/napcat_voice_config.json")
        if path.exists():
            try:
                config.update(json.loads(path.read_text()))
                logger.info("napcat_voice loaded config: %s", config)
            except Exception as exc:
                logger.warning("napcat_voice config load failed: %s", exc)
        else:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
                logger.info("napcat_voice wrote default config to %s", path)
            except Exception as exc:
                logger.warning("napcat_voice config write failed: %s", exc)
        return config

    # ── NapCat action helpers ──

    async def _call_napcat(self, event: AstrMessageEvent, action: str, **params) -> Any:
        fn = getattr(event, "call_napcat_action", None)
        if not callable(fn):
            raise RuntimeError("当前平台不支持 NapCat action，请确认使用 aiocqhttp 适配器。")
        return await fn(action=action, **params)

    def _session_key(self, event: AstrMessageEvent) -> str:
        return getattr(event, "unified_msg_origin", str(event.get_sender_id()))

    def _resolve_user_id(self, event: AstrMessageEvent, user_id: str = "") -> str:
        if user_id:
            return str(user_id)
        key = self._session_key(event)
        return self._active_call_user.get(key, str(event.get_sender_id()))

    # ── provider lookup ──

    def _get_stt(self, event: AstrMessageEvent):
        prov = (self.context.get_using_stt_provider(event.unified_msg_origin)
                or self.context.get_using_stt_provider()
                or getattr(self.context.provider_manager, "curr_stt_provider_inst", None))
        if not prov and self.context.provider_manager.stt_provider_insts:
            prov = self.context.provider_manager.stt_provider_insts[0]
        return prov

    def _get_tts(self, event: AstrMessageEvent):
        prov = (self.context.get_using_tts_provider(event.unified_msg_origin)
                or self.context.get_using_tts_provider()
                or getattr(self.context.provider_manager, "curr_tts_provider_inst", None))
        if not prov and self.context.provider_manager.tts_provider_insts:
            prov = self.context.provider_manager.tts_provider_insts[0]
        return prov

    # ── auto loop management ──

    def _has_running_auto_loop(self, key: str) -> bool:
        t = self._auto_loop_tasks.get(key)
        return bool(t and not t.done())

    async def _stop_auto_loop(self, key: str) -> None:
        task = self._auto_loop_tasks.pop(key, None)
        self._auto_loop_drain_before_turn.pop(key, None)
        self._auto_loop_resume_after_playback.pop(key, None)
        self._auto_loop_connected_context_pending.pop(key, None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ── call status ──

    @staticmethod
    def _extract_data(response: Any) -> dict[str, Any]:
        payload = response
        if isinstance(payload, dict):
            for k in ("data", "response", "result"):
                if k in payload and payload[k] is not None:
                    payload = payload[k]
                    break
        if isinstance(payload, list):
            return {"items": payload}
        if not isinstance(payload, dict):
            return {}
        return {k: v for k, v in payload.items()
                if k not in {"status", "retcode", "message", "wording", "echo", "action", "seq"}}

    @staticmethod
    def _is_action_ok(response: Any) -> bool:
        if not isinstance(response, dict):
            return bool(response)
        for key in ("ok", "success"):
            if response.get(key) is True:
                return True
        if str(response.get("status", "")).lower() in {"ok", "done", "success"}:
            return True
        if response.get("retcode") in {0, "0"}:
            return True
        markers = ("sent_count", "frame_count", "room_id", "active", "peer_connected", "state")
        return any(k in response for k in markers)

    def _get_call_status(self, response: Any) -> dict[str, Any]:
        data = self._extract_data(response)
        state = str(data.get("state", "")).upper()
        peer = bool(data.get("peer_connected"))
        active = bool(data.get("active"))
        connected = peer or state in {"CONNECTED", "STREAMING"}
        return {
            "data": data, "state": state, "peer_connected": peer,
            "active": active, "connected": connected,
            "active_call": active or connected or state == "RINGING",
        }

    async def _wait_for_connected(
        self, event: AstrMessageEvent, user_id: str,
        timeout: int = 60, interval: float = 0.8,
    ) -> dict[str, Any]:
        start = time.monotonic()
        consecutive_errors = 0
        while time.monotonic() - start < timeout:
            try:
                resp = await self._call_napcat(event, "voice_call_query", user_id=user_id)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    return {"ok": False, "connected": False, "reason": "query_failed"}
                await asyncio.sleep(interval)
                continue
            status = self._get_call_status(resp)
            if status["connected"]:
                return {"ok": True, "connected": True, "data": status["data"]}
            if not status["active_call"]:
                return {"ok": False, "connected": False, "reason": "call_dropped"}
            await asyncio.sleep(interval)
        return {"ok": False, "connected": False, "reason": "timeout"}

    # ── VAD ──

    _SILENCE_OPUS_BYTES = 10  # Opus frames <= this size are silence/DTX

    @staticmethod
    def _frame_energy(frame: dict[str, Any]) -> int:
        """Estimate frame energy. Uses Opus payload size (bytes) as proxy.

        Opus encodes silence/low-energy in very few bytes (~3-10),
        while speech typically needs 20-200+ bytes. This is far more
        reliable than PCM peak when NapCat's decode is broken.
        """
        opus = frame.get("opus_data_hex")
        if isinstance(opus, str) and opus:
            return len(opus) // 2  # byte count
        return 0

    async def _collect_utterance(
        self, event: AstrMessageEvent, user_id: str,
        max_frames: int, wait_seconds: float, poll_interval: float,
        min_voiced: int, start_thresh: int, stop_thresh: int,
        silence_frames_needed: int, trace_id: str = "",
    ) -> dict[str, Any]:
        cfg = self._vad_config
        min_utt_ms = int(cfg.get("min_utterance_ms", 400))
        post_frames = int(cfg.get("post_speech_frames", 5))
        pre_roll_size = int(cfg.get("pre_roll_size", 16))
        adaptive = bool(cfg.get("adaptive_silence", True))

        t0 = time.monotonic()
        deadline = t0 + max(0.0, wait_seconds)
        ts_ms = int(time.time() * 1000)
        payload = {"user_id": user_id, "max_frames": max(50, max_frames), "peek": False, "decode": True}

        pre_roll: deque[dict[str, Any]] = deque(maxlen=max(8, pre_roll_size))
        frames: list[dict[str, Any]] = []
        speech = False
        speech_at = 0.0
        voiced = 0
        silence = 0
        cand_hits = 0
        cand_peak = 0
        cur_start = start_thresh
        cur_stop = stop_thresh
        confirmed = False

        total_received = 0
        energy_samples: list[int] = []
        while True:
            resp = await self._call_napcat(event, "voice_call_recv_audio", **payload)
            resp_ok = isinstance(resp, dict) and self._is_action_ok(resp)
            if resp_ok:
                data = self._extract_data(resp)
                raw_frames = data.get("frames")
                if isinstance(raw_frames, list):
                    for fr in raw_frames:
                        if not isinstance(fr, dict):
                            continue
                        total_received += 1
                        peak = self._frame_energy(fr)
                        if len(energy_samples) < 50:
                            energy_samples.append(peak)
                        if speech:
                            frames.append(fr)
                            if peak >= cur_stop:
                                voiced += 1
                                silence = max(0, silence - 3)
                            else:
                                silence += 1
                        else:
                            pre_roll.append(fr)
                            if peak >= cur_start:
                                cand_hits += 1
                                cand_peak = max(cand_peak, peak)
                                if cand_hits >= 2:
                                    speech = True
                                    speech_at = time.monotonic()
                                    frames.extend(list(pre_roll))
                                    voiced = max(2, cand_hits)
                                    silence = 0
                                    if trace_id:
                                        logger.info(
                                            "napcat_voice trace=%s stage=vad.speech_start fields=%s",
                                            trace_id,
                                            json.dumps({"peak": cand_peak, "start_threshold": cur_start,
                                                        "stop_threshold": cur_stop, "candidate_hits": cand_hits}, ensure_ascii=False),
                                        )
                            else:
                                cand_hits = 0
                                cand_peak = 0

                    # end-of-speech check
                    if speech and voiced >= max(1, min_voiced):
                        dur_ms = int((time.monotonic() - speech_at) * 1000)
                        eff_silence = silence_frames_needed
                        if adaptive and dur_ms > 1000:
                            eff_silence = int(silence_frames_needed * min(2.0, 1.0 + (dur_ms - 1000) / 4000))
                        if silence >= max(3, eff_silence) and dur_ms >= min_utt_ms:
                            if post_frames > 0 and not confirmed:
                                confirmed = True
                                continue
                            elapsed = int((time.monotonic() - t0) * 1000)
                            if trace_id:
                                logger.info(
                                    "napcat_voice trace=%s stage=turn.vad.ok fields=%s",
                                    trace_id,
                                    json.dumps({"voiced_frames": voiced, "silence_frames": silence,
                                                "frame_count": len(frames), "speech_duration_ms": dur_ms,
                                                "listen_elapsed_ms": elapsed}, ensure_ascii=False),
                                )
                            return {"ok": True, "frames": frames, "frame_count": len(frames),
                                    "voiced_frames": voiced, "silence_frames": silence,
                                    "listen_elapsed_ms": elapsed}

            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(max(0.05, poll_interval))

        # timeout — return what we have
        if speech and voiced >= max(1, min_voiced) and frames:
            return {"ok": True, "frames": frames, "frame_count": len(frames),
                    "voiced_frames": voiced, "silence_frames": silence,
                    "listen_elapsed_ms": int((time.monotonic() - t0) * 1000)}
        if trace_id:
            logger.info(
                "napcat_voice trace=%s stage=turn.vad.failed fields=%s",
                trace_id,
                json.dumps({"reason": "no_speech_detected", "voiced_frames": voiced,
                            "silence_frames": silence, "frame_count": len(frames),
                            "total_received": total_received, "cur_start": cur_start,
                            "energy_samples": energy_samples[:30],
                            "listen_elapsed_ms": int((time.monotonic() - t0) * 1000)}, ensure_ascii=False),
            )
        return {"ok": False, "reason": "no_speech_detected", "voiced_frames": voiced,
                "silence_frames": silence, "frame_count": len(frames)}

    # ── PCM → WAV (with normalization) ──

    @classmethod
    def _opus_frames_to_ogg(cls, frames: list[Any]) -> bytes:
        """Build a minimal OGG/Opus container from raw Opus packets.

        Filters out silence/DTX frames to keep only voiced audio.
        """
        packets: list[bytes] = []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            raw = fr.get("opus_data_hex")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                pkt = bytes.fromhex(raw)
            except ValueError:
                continue
            # skip silence frames
            if len(pkt) <= cls._SILENCE_OPUS_BYTES:
                continue
            packets.append(pkt)
        if not packets:
            return b""

        serial = 0x41535442  # "ASTB"
        granule = 0

        def _ogg_page(page_seq: int, pkt_data: bytes, header_type: int = 0,
                       gran: int = 0) -> bytes:
            segs = []
            off = 0
            while off < len(pkt_data):
                chunk = min(255, len(pkt_data) - off)
                segs.append(chunk)
                off += chunk
            if not segs or segs[-1] == 255:
                segs.append(0)
            seg_table = bytes(segs)
            hdr = bytearray(27 + len(seg_table) + len(pkt_data))
            hdr[0:4] = b"OggS"
            hdr[4] = 0  # version
            hdr[5] = header_type
            struct.pack_into("<Q", hdr, 6, gran & 0xFFFFFFFFFFFFFFFF)
            struct.pack_into("<I", hdr, 14, serial)
            struct.pack_into("<I", hdr, 18, page_seq)
            hdr[26] = len(seg_table)
            hdr[27:27 + len(seg_table)] = seg_table
            hdr[27 + len(seg_table):] = pkt_data
            # CRC
            crc = 0
            for b in hdr:
                crc = ((crc << 8) ^ _OGG_CRC_TABLE[(crc >> 24) ^ b]) & 0xFFFFFFFF
            struct.pack_into("<I", hdr, 22, crc)
            return bytes(hdr)

        # OpusHead
        opus_head = struct.pack("<8sBBHIhB", b"OpusHead", 1, 1, 0, 48000, 0, 0)
        # OpusTags
        vendor = b"AstrBot"
        opus_tags = struct.pack("<8sI", b"OpusTags", len(vendor)) + vendor + struct.pack("<I", 0)

        pages = bytearray()
        pages += _ogg_page(0, opus_head, header_type=2, gran=0)
        pages += _ogg_page(1, opus_tags, gran=0)

        # Audio pages — batch packets into pages
        page_seq = 2
        samples_per_frame = 960  # 20ms at 48kHz
        for pkt in packets:
            granule += samples_per_frame
            pages += _ogg_page(page_seq, pkt, gran=granule)
            page_seq += 1

        # Mark last page as EOS
        if pages:
            # Re-write last page with EOS flag
            last_start = bytes(pages).rfind(b"OggS")
            if last_start >= 0:
                pages[last_start + 5] |= 0x04

        return bytes(pages)

    @staticmethod
    def _frames_to_wav(frames: list[Any], wav_path: str) -> str:
        """Convert decoded PCM frames to normalized 16kHz mono WAV."""
        pcm_buf = bytearray()
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            raw = fr.get("pcm_data_hex")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                chunk = bytes.fromhex(raw)
            except ValueError:
                continue
            if all(b == 0 for b in chunk):
                continue
            pcm_buf.extend(chunk)

        if len(pcm_buf) < 640:
            return ""

        # Normalize volume
        n = len(pcm_buf) // 2
        samples = list(struct.unpack(f"<{n}h", bytes(pcm_buf)))
        mx = max(abs(s) for s in samples)
        if mx < 300:
            # Too quiet — pure noise, skip
            return ""
        if mx > 0:
            gain = min(28000 / mx, 50.0)
            if gain > 1.05:
                samples = [max(-32768, min(32767, int(s * gain))) for s in samples]
                pcm_buf = bytearray(struct.pack(f"<{n}h", *samples))
                logger.debug("napcat_voice PCM normalized: peak %d -> %d (gain=%.1f)",
                             mx, int(mx * gain), gain)

        # Write WAV directly - NapCat PCM is 16kHz mono s16le
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm_buf)
        return wav_path

    # ── STT ──

    _STT_OVERRIDES = {
        "language": "zh",
        "prompt": "转录中文语音",
        "system_prompt": "你是语音转录助手。将音频中的语音内容准确转录为简体中文，仅返回转录文本。不要翻译，不要猜测，不要输出英文。如果听不清则返回空字符串。",
        "user_prompt": "请将音频中的语音内容转录为简体中文文本，仅返回转录内容。",
    }

    async def _run_stt(self, event: AstrMessageEvent, wav_path: str, retry: int = 4, trace_id: str = "") -> str:
        prov = self._get_stt(event)
        if not prov:
            raise RuntimeError("未配置 STT Provider")

        last_err: Exception | None = None
        for attempt in range(1, max(1, retry) + 1):
            try:
                if not Path(wav_path).exists():
                    raise FileNotFoundError(wav_path)
                # Lock serializes voice STT calls and protects the
                # temporary attribute overrides on the shared provider.
                async with self._stt_lock:
                    orig: dict[str, Any] = {}
                    for attr, val in self._STT_OVERRIDES.items():
                        if hasattr(prov, attr):
                            orig[attr] = getattr(prov, attr)
                            setattr(prov, attr, val)
                    try:
                        text = await prov.get_text(audio_url=wav_path)
                    finally:
                        for attr, val in orig.items():
                            setattr(prov, attr, val)
                    if trace_id:
                        logger.info("napcat_voice trace=%s stage=stt.ok fields=%s",
                                    trace_id, json.dumps({"attempt": attempt, "recognized_text": text}, ensure_ascii=False))
                    return text.strip()
            except FileNotFoundError:
                last_err = FileNotFoundError(wav_path)
                await asyncio.sleep(0.5)
            except Exception as e:
                last_err = e
                if trace_id:
                    logger.warning("napcat_voice trace=%s stage=stt.error fields=%s",
                                   trace_id, json.dumps({"attempt": attempt, "error": str(e)[:200]}, ensure_ascii=False))
                await asyncio.sleep(0.2)
        if last_err and "empty transcription" in str(last_err).lower():
            return ""
        if last_err:
            raise last_err
        return ""

    @staticmethod
    def _is_non_speech(text: str) -> bool:
        n = text.strip().lower()
        if not n:
            return True
        n = re.sub(r"\s+", " ", n)
        canned = {"no speech detected in the audio.", "no speech detected", "silence", "empty audio",
                   "uh-huh", "uh huh", "huh", "uh", "hmm", "hm", "mhm", "um", "umm", "erm", "ah", "eh"}
        if n.rstrip(" .!?,") in canned:
            return True
        return bool(re.fullmatch(
            r"(?:um+|uh+|hmm+|hm+|erm|ah|eh|well|mhm|mm-hmm|mm hmm)"
            r"(?:[\s,.\-!?]+(?:um+|uh+|hmm+|hm+|erm|ah|eh|well|mhm|mm-hmm|mm hmm))*"
            r"[\s,.\-!?]*", n))

    # ── TTS → play into call ──

    async def _run_tts(self, event: AstrMessageEvent, text: str) -> str | None:
        prov = self._get_tts(event)
        if not prov:
            return None
        try:
            path = await asyncio.wait_for(prov.get_audio(text=text), timeout=20.0)
            return str(path) if path and Path(path).exists() else None
        except Exception as exc:
            logger.warning("napcat_voice TTS failed: %s", exc)
            return None

    @staticmethod
    def _ogg_to_opus_frames(ogg_path: str) -> list[str]:
        data = Path(ogg_path).read_bytes()
        packets: list[bytes] = []
        buf = bytearray()
        off = 0
        while off + 27 <= len(data):
            if data[off:off+4] != b"OggS":
                break
            segs = data[off + 26]
            tbl_end = off + 27 + segs
            if tbl_end > len(data):
                break
            tbl = data[off+27:tbl_end]
            pay_end = tbl_end + sum(tbl)
            if pay_end > len(data):
                break
            p = tbl_end
            for lace in tbl:
                buf.extend(data[p:p+lace])
                p += lace
                if lace < 255:
                    packets.append(bytes(buf))
                    buf.clear()
            off = pay_end
        return [pkt.hex() for pkt in packets if pkt and not pkt.startswith((b"OpusHead", b"OpusTags"))]

    async def _play_wav_in_call(self, event: AstrMessageEvent, user_id: str, wav_path: str) -> bool:
        ogg_path: str | None = None
        try:
            ogg_path = await convert_audio_format(wav_path, "ogg")
            opus = await asyncio.to_thread(self._ogg_to_opus_frames, ogg_path)
            if not opus:
                return False
            for i in range(0, len(opus), 120):
                chunk = opus[i:i+120]
                resp = await self._call_napcat(event, "voice_call_send_audio", user_id=user_id, frames=chunk)
                if not self._is_action_ok(resp):
                    return False
            return True
        except Exception as exc:
            logger.warning("napcat_voice play_wav failed: %s", exc)
            return False
        finally:
            if ogg_path and Path(ogg_path).exists():
                try:
                    Path(ogg_path).unlink()
                except OSError:
                    pass

    async def _reply_in_call(self, event: AstrMessageEvent, text: str, user_id: str, trace_id: str = "") -> bool:
        """TTS text → play opus frames into active call."""
        # check call still active
        try:
            resp = await self._call_napcat(event, "voice_call_query", user_id=user_id)
            st = self._get_call_status(resp)
            if not st["active_call"]:
                return False
        except Exception:
            return False

        wav_path = await self._run_tts(event, text)
        if not wav_path:
            logger.warning("napcat_voice TTS returned no audio for: %s", text[:60])
            return False

        ok = await self._play_wav_in_call(event, user_id, wav_path)
        if trace_id:
            logger.info("napcat_voice trace=%s stage=turn.playback fields=%s",
                        trace_id, json.dumps({"ok": ok, "wav_path": wav_path}, ensure_ascii=False))
        return ok

    # ── drain buffer ──

    async def _drain_buffer(self, event: AstrMessageEvent, user_id: str, max_frames: int = 200) -> int:
        drained = 0
        for _ in range(6):
            resp = await self._call_napcat(event, "voice_call_recv_audio",
                                           user_id=user_id, max_frames=max_frames, peek=False, decode=False)
            if not isinstance(resp, dict) or not self._is_action_ok(resp):
                break
            n = int(self._extract_data(resp).get("frame_count", 0) or 0)
            if n <= 0:
                break
            drained += n
            if n < max_frames:
                break
        return drained

    # ══════════════════════════════════════════════════════════════
    # Main auto-loop
    # ══════════════════════════════════════════════════════════════

    async def _run_auto_loop(
        self, event: AstrMessageEvent, session_key: str, user_id: str,
        cfg: dict[str, Any], welcome_text: str = "",
    ) -> None:
        logger.info("napcat_voice auto loop started: session=%s user=%s", session_key, user_id)
        call_ended = False
        pipeline_task: asyncio.Task[Any] | None = None

        max_frames = int(cfg.get("max_frames", 150))
        wait_s = float(cfg.get("wait_seconds", 8.0))
        poll = float(cfg.get("poll_interval", 0.1))
        min_voiced = int(cfg.get("min_non_silent_frames", 3))
        start_th = int(cfg.get("vad_start_threshold", 20))
        stop_th = int(cfg.get("vad_stop_threshold", 12))
        silence_fr = int(cfg.get("vad_silence_frames", 15))

        try:
            # welcome (retry up to 3 times since TTS may need warmup)
            if welcome_text:
                for _welcome_try in range(3):
                    try:
                        ok = await self._reply_in_call(event, welcome_text, user_id)
                        if ok:
                            self._auto_loop_resume_after_playback[session_key] = time.monotonic() + 2.0
                            self._auto_loop_drain_before_turn[session_key] = True
                            break
                        await asyncio.sleep(1.0)
                    except Exception as exc:
                        logger.warning("napcat_voice welcome failed (attempt %d): %s", _welcome_try + 1, exc)
                        await asyncio.sleep(1.0)

            query_errors = 0
            while True:
                # check call alive
                try:
                    resp = await self._call_napcat(event, "voice_call_query", user_id=user_id)
                    query_errors = 0
                    st = self._get_call_status(resp)
                    if not st["active_call"]:
                        call_ended = True
                        break
                    if not st["connected"]:
                        await asyncio.sleep(0.8)
                        continue
                except Exception:
                    query_errors += 1
                    if query_errors >= 10:
                        logger.warning("napcat_voice auto loop: query failed %d times, exiting", query_errors)
                        call_ended = True
                        break
                    await asyncio.sleep(1.0)
                    continue

                # playback cooldown
                resume = self._auto_loop_resume_after_playback.get(session_key, 0.0)
                remaining = resume - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(remaining, 3.0))
                    self._auto_loop_drain_before_turn[session_key] = True
                    continue

                if self._auto_loop_drain_before_turn.pop(session_key, False):
                    await self._drain_buffer(event, user_id, max(200, max_frames * 2))

                # VAD
                trace_id = f"{session_key}:{uuid.uuid4().hex[:6]}"
                utterance = await self._collect_utterance(
                    event, user_id, max_frames, wait_s, poll,
                    min_voiced, start_th, stop_th, silence_fr, trace_id,
                )
                if not utterance.get("ok"):
                    await asyncio.sleep(0.1)
                    continue

                # build wav from PCM frames
                wav_path = self._frames_to_wav(
                    utterance.get("frames", []),
                    str(self._tts_cache_dir / f"recv_{time.time_ns()}.wav"),
                )
                if not wav_path:
                    continue

                # fire async pipeline (don't wait — VAD continues collecting)
                connected_ctx = self._auto_loop_connected_context_pending.pop(session_key, False)
                pipeline_task = asyncio.create_task(
                    self._pipeline(event, session_key, user_id, wav_path, trace_id, connected_ctx),
                    name=f"napcat_voice_pipeline:{session_key}",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("napcat_voice auto loop crashed: %s", exc)
        finally:
            if pipeline_task and not pipeline_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=15.0)
                except Exception:
                    pass
            if asyncio.current_task() is self._auto_loop_tasks.get(session_key):
                self._auto_loop_tasks.pop(session_key, None)
            self._auto_loop_drain_before_turn.pop(session_key, None)
            self._auto_loop_connected_context_pending.pop(session_key, None)
            self._active_call_user.pop(session_key, None)
            self._call_locks.pop(session_key, None)
            logger.info("napcat_voice auto loop finished: session=%s ended=%s", session_key, call_ended)

    async def _pipeline(
        self, event: AstrMessageEvent, session_key: str, user_id: str,
        wav_path: str, trace_id: str, connected_ctx: bool,
    ) -> None:
        """STT → AstrBot core → TTS → play (runs async)."""
        try:
            # STT
            text = (await self._run_stt(event, wav_path, retry=4, trace_id=trace_id)).strip()
            if not text or self._is_non_speech(text):
                return

            # Push clean text through AstrBot core pipeline
            if not isinstance(event, AiocqhttpMessageEvent):
                return
            synth = _SyntheticVoiceEvent(
                source_event=event,
                recognized_text=text,
                injected_text=text,
                forward_to_chat=True,
            )
            synth.set_extra("napcat_voice_call_active", True)
            if connected_ctx:
                synth.set_extra("napcat_voice_just_connected", True)
            synth.plugins_name = [
                p for p in (event.plugins_name or []) if p != "napcat-voice"
            ] or ["astrbot", "web_searcher", "builtin_commands", "session_controller"]
            self.context.get_event_queue().put_nowait(synth)

            reply = await synth.wait_for_reply(timeout=90.0)
            if not reply:
                return

            logger.info("napcat_voice trace=%s stage=turn.reply.generated fields=%s",
                        trace_id, json.dumps({"reply_text": reply[:120]}, ensure_ascii=False))

            # Block VAD during TTS playback to avoid recording bot's own voice
            self._auto_loop_resume_after_playback[session_key] = time.monotonic() + 60.0
            # TTS → play
            ok = await self._reply_in_call(event, reply, user_id, trace_id)
            # After playback, set short cooldown for remaining echo to fade
            self._auto_loop_resume_after_playback[session_key] = time.monotonic() + 1.5
            self._auto_loop_drain_before_turn[session_key] = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("napcat_voice pipeline failed: session=%s err=%s", session_key, exc)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ── start auto loop ──

    async def _start_auto_loop(self, event: AstrMessageEvent, user_id: str,
                                cfg: dict[str, Any], welcome_text: str = "") -> dict[str, Any]:
        # pre-flight
        missing: list[str] = []
        if not self._get_stt(event):
            missing.append("STT")
        if not self._get_tts(event):
            missing.append("TTS")
        if missing:
            return {"ok": False, "reason": "missing_providers", "missing": missing,
                    "error": f"语音通话需要 {', '.join(missing)} 服务。"}

        key = self._session_key(event)
        await self._stop_auto_loop(key)
        self._auto_loop_drain_before_turn[key] = True
        self._auto_loop_connected_context_pending[key] = True
        task = asyncio.create_task(
            self._run_auto_loop(event, key, user_id, cfg, welcome_text),
            name=f"napcat_voice_loop:{key}",
        )
        self._auto_loop_tasks[key] = task
        return {"ok": True, "status": "started"}

    # ══════════════════════════════════════════════════════════════
    # LLM Tools (only the essentials)
    # ══════════════════════════════════════════════════════════════

    @llm_tool(name="napcat_voice_call_start")
    async def voice_call_start(
        self, event: AstrMessageEvent,
        user_id: str = "",
    ) -> str:
        """Start a voice call with the user. Initiates call, waits for answer, then runs automatic voice loop.

        Args:
            user_id(string): target user QQ id. Leave empty to call the current user.
        """
        if not event.is_admin():
            return "权限不足，仅管理员可发起语音通话。"
        return await self._do_handsfree(event, user_id)

    @llm_tool(name="napcat_voice_call_query")
    async def voice_call_query(self, event: AstrMessageEvent, user_id: str = "") -> str:
        """Query voice call state.

        Args:
            user_id(string): target user id. Leave empty to use current session user.
        """
        if not event.is_admin():
            return "权限不足，仅管理员可查询语音通话状态。"
        uid = self._resolve_user_id(event, user_id)
        try:
            resp = await self._call_napcat(event, "voice_call_query", user_id=uid)
            return json.dumps(resp, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    @llm_tool(name="napcat_voice_leave")
    async def voice_leave(self, event: AstrMessageEvent, user_id: str = "") -> str:
        """Hang up the current voice call.

        Args:
            user_id(string): target user id. Leave empty to use current session user.
        """
        if not event.is_admin():
            return "权限不足，仅管理员可挂断语音通话。"
        key = self._session_key(event)
        uid = self._resolve_user_id(event, user_id)
        await self._stop_auto_loop(key)
        self._active_call_user.pop(key, None)
        try:
            resp = await self._call_napcat(event, "voice_call_leave", user_id=uid)
            return json.dumps({"ok": self._is_action_ok(resp), "response": resp}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    # ══════════════════════════════════════════════════════════════
    # Handsfree entry point (used by LLM tool + regex trigger)
    # ══════════════════════════════════════════════════════════════

    async def _do_handsfree(self, event: AstrMessageEvent, user_id: str = "") -> str:
        self._vad_config = self._load_vad_config()
        cfg = self._vad_config
        uid = self._resolve_user_id(event, user_id)
        key = self._session_key(event)

        # per-session lock to prevent concurrent call attempts
        lock = self._call_locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            return json.dumps({"ok": False, "reason": "call_in_progress"}, ensure_ascii=False)

        async with lock:
            return await self._do_handsfree_inner(event, uid, key, cfg)

    async def _do_handsfree_inner(
        self, event: AstrMessageEvent, uid: str, key: str, cfg: dict[str, Any],
    ) -> str:
        # already in call?
        if self._has_running_auto_loop(key):
            return json.dumps({"ok": False, "reason": "already_in_call"}, ensure_ascii=False)

        # clean up any stale state
        self._active_call_user.pop(key, None)

        try:
            start_resp = await self._call_napcat(event, "voice_call_start", user_id=uid)
        except Exception as e:
            return json.dumps({"ok": False, "reason": "call_start_failed", "error": str(e)}, ensure_ascii=False)
        if not self._is_action_ok(start_resp):
            return json.dumps({"ok": False, "reason": "call_start_failed"}, ensure_ascii=False)

        self._active_call_user[key] = uid

        # wait for connection
        conn = await self._wait_for_connected(event, uid, timeout=60, interval=0.8)
        if not conn.get("connected"):
            self._active_call_user.pop(key, None)
            # 主动挂断，清理残留通话
            try:
                await self._call_napcat(event, "voice_call_leave", user_id=uid)
            except Exception:
                pass
            return json.dumps({"ok": False, "reason": conn.get("reason", "not_connected")}, ensure_ascii=False)

        # start auto loop
        welcome = cfg.get("welcome_text", "我已接通，你可以直接说话。")
        loop_result = await self._start_auto_loop(event, uid, cfg, welcome)

        return json.dumps({
            "ok": loop_result.get("ok", False),
            "status": "handsfree_started" if loop_result.get("ok") else "loop_failed",
            "target_user_id": uid,
            **({k: v for k, v in loop_result.items() if k not in ("ok",)}),
        }, ensure_ascii=False)

    # ══════════════════════════════════════════════════════════════
    # Text command handlers
    # ══════════════════════════════════════════════════════════════

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.regex(r"^(给我)?打(个|电)?(电话|来)$")
    async def on_call_keyword(self, event: AstrMessageEvent) -> None:
        if not isinstance(event, AiocqhttpMessageEvent):
            event.set_result(event.plain_result("语音通话仅支持 aiocqhttp 平台。").use_t2i(False))
            return
        # pre-flight
        missing = []
        if not self._get_stt(event):
            missing.append("STT")
        if not self._get_tts(event):
            missing.append("TTS")
        if missing:
            event.set_result(event.plain_result(f"需要 {'、'.join(missing)} 服务。").use_t2i(False))
            return
        event.stop_event()
        event.set_result(event.plain_result("正在发起语音通话，请稍候接听...").use_t2i(False))
        asyncio.create_task(self._keyword_call(event), name=f"napcat_call:{self._session_key(event)}")

    async def _keyword_call(self, event: AstrMessageEvent) -> None:
        try:
            await self._do_handsfree(event)
        except Exception as exc:
            logger.error("napcat_voice keyword call failed: %s", exc)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.regex(r"^挂(断|了|掉)?(电话)?$")
    async def on_hangup_keyword(self, event: AstrMessageEvent) -> None:
        if not isinstance(event, AiocqhttpMessageEvent):
            return
        event.stop_event()
        key = self._session_key(event)
        if not self._has_running_auto_loop(key) and key not in self._active_call_user:
            event.set_result(event.plain_result("当前没有进行中的语音通话。").use_t2i(False))
            return
        uid = self._resolve_user_id(event)
        await self._stop_auto_loop(key)
        self._active_call_user.pop(key, None)
        try:
            await self._call_napcat(event, "voice_call_leave", user_id=uid)
        except Exception:
            pass
        event.set_result(event.plain_result("已挂断语音通话。").use_t2i(False))

    @filter.on_llm_request()
    async def on_voice_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Inject voice-call context into system_prompt for voice events."""
        if not event.get_extra("napcat_voice_call_active"):
            return
        ctx = (
            "\n\n[语音通话模式] 你正在和用户进行实时语音通话。"
            "请用简短、自然、适合语音播报的中文回复。"
            "不要发起、接听、查询、重拨或挂断电话。"
            "不要提及工具名称、工具参数或工具结果。"
            "不要输出 Markdown 格式。"
        )
        if event.get_extra("napcat_voice_just_connected"):
            ctx = "\n\n[电话刚刚接通]" + ctx
        req.system_prompt = (req.system_prompt or "") + ctx

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("vad_config")
    async def vad_config_cmd(self, event: AstrMessageEvent, action: str = "") -> None:
        """Show or reload VAD config. Usage: /vad_config [reload]"""
        if action == "reload":
            self._vad_config = self._load_vad_config()
            event.set_result(event.plain_result(
                f"VAD 配置已重载:\n{json.dumps(self._vad_config, ensure_ascii=False, indent=2)}"
            ).use_t2i(False))
        else:
            event.set_result(event.plain_result(
                f"当前 VAD 配置:\n{json.dumps(self._vad_config, ensure_ascii=False, indent=2)}"
            ).use_t2i(False))

    async def terminate(self) -> None:
        for key in list(self._auto_loop_tasks):
            await self._stop_auto_loop(key)
        self._active_call_user.clear()

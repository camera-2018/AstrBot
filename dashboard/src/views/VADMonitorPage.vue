<template>
  <v-container fluid class="pa-4">
    <!-- Header -->
    <v-row align="center" class="mb-4">
      <v-col cols="auto">
        <h2 class="text-h5 font-weight-bold">
          <v-icon class="mr-2">mdi-waveform</v-icon>
          VAD Monitor
        </h2>
      </v-col>
      <v-col cols="auto">
        <v-chip :color="statusColor" variant="flat" size="small" class="font-weight-bold">
          {{ statusLabel }}
        </v-chip>
      </v-col>
      <v-spacer />
      <v-col cols="auto">
        <v-btn size="small" variant="outlined" color="primary" @click="loadConfig" class="mr-2">
          <v-icon start>mdi-refresh</v-icon>Reload
        </v-btn>
        <v-btn size="small" variant="outlined" color="error" @click="clearAll">
          <v-icon start>mdi-delete</v-icon>Clear
        </v-btn>
      </v-col>
    </v-row>

    <!-- Config Controls -->
    <v-card class="mb-4" variant="outlined">
      <v-card-text class="py-2">
        <v-row align="center" dense>
          <v-col v-for="p in configFields" :key="p.key" cols="auto">
            <v-text-field
              v-model.number="config[p.key]"
              :label="p.label"
              type="number"
              density="compact"
              variant="outlined"
              hide-details
              style="width: 130px"
              :step="p.step || 1"
            />
          </v-col>
          <v-col cols="auto">
            <v-btn color="primary" size="small" @click="saveConfig" :loading="saving">
              <v-icon start>mdi-content-save</v-icon>Save
            </v-btn>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-row>
      <!-- Waveform -->
      <v-col cols="12" md="8">
        <v-card variant="outlined" class="fill-height">
          <v-card-title class="text-subtitle-2 pb-0">Peak Amplitude</v-card-title>
          <v-card-text class="pa-2">
            <canvas ref="waveCanvas" class="waveform-canvas" />
            <!-- Peak bar -->
            <div class="peak-bar mt-2">
              <div class="peak-fill" :style="{ width: peakPct + '%' }" />
              <div class="threshold-line start-line" :style="{ left: startThreshPct + '%' }" />
              <div class="threshold-line stop-line" :style="{ left: stopThreshPct + '%' }" />
              <span class="peak-label">{{ currentPeak }}</span>
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Stats -->
      <v-col cols="12" md="4">
        <v-card variant="outlined" class="fill-height">
          <v-card-title class="text-subtitle-2 pb-0">Live Stats</v-card-title>
          <v-card-text>
            <v-row dense>
              <v-col v-for="s in stats" :key="s.key" cols="6">
                <div class="text-center pa-2 rounded stat-box">
                  <div class="text-h5 font-weight-bold text-primary">{{ s.value }}</div>
                  <div class="text-caption text-medium-emphasis">{{ s.label }}</div>
                </div>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Event Log -->
    <v-card class="mt-4" variant="outlined">
      <v-card-title class="text-subtitle-2 pb-0">Event Log</v-card-title>
      <v-card-text class="pa-2">
        <div ref="logArea" class="log-area">
          <div
            v-for="(entry, i) in logEntries"
            :key="i"
            :class="'log-' + entry.level"
            class="log-line"
          >
            [{{ entry.time }}] {{ entry.text }}
          </div>
        </div>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import axios from 'axios'

const MAX_PEAK = 100
const MAX_WAVE = 300
const POLL_MS = 400

// --- State ---
const status = ref<'idle' | 'listening' | 'speech' | 'tts'>('idle')
const currentPeak = ref(0)
const waveData = ref<{ peak: number; voiced: boolean }[]>([])
const logEntries = ref<{ time: string; text: string; level: string }[]>([])
const saving = ref(false)
const logArea = ref<HTMLDivElement>()
const waveCanvas = ref<HTMLCanvasElement>()

const statsData = reactive({
  voiced: 0,
  silence: 0,
  peak: 0,
  turns: 0,
  stt_ms: '-' as string | number,
  tts_ms: '-' as string | number,
})

const config: Record<string, any> = reactive({
  vad_start_threshold: 20,
  vad_stop_threshold: 12,
  vad_silence_frames: 15,
  min_non_silent_frames: 3,
  wait_seconds: 8,
  poll_interval: 0.1,
  max_frames: 150,
  welcome_text: '',
})

const configFields = [
  { key: 'vad_start_threshold', label: 'Start Thresh', step: 1 },
  { key: 'vad_stop_threshold', label: 'Stop Thresh', step: 1 },
  { key: 'vad_silence_frames', label: 'Silence Frames', step: 1 },
  { key: 'min_non_silent_frames', label: 'Min Voiced', step: 1 },
  { key: 'wait_seconds', label: 'Wait (s)', step: 0.5 },
  { key: 'poll_interval', label: 'Poll (s)', step: 0.01 },
  { key: 'min_utterance_ms', label: 'Min Utt (ms)', step: 50 },
  { key: 'post_speech_frames', label: 'Post Speech', step: 1 },
  { key: 'pre_roll_size', label: 'Pre Roll', step: 1 },
]

const stats = computed(() => [
  { key: 'voiced', label: 'voiced frames', value: statsData.voiced },
  { key: 'silence', label: 'silence frames', value: statsData.silence },
  { key: 'peak', label: 'current peak', value: statsData.peak },
  { key: 'turns', label: 'turns done', value: statsData.turns },
  { key: 'stt', label: 'last STT (ms)', value: statsData.stt_ms },
  { key: 'tts', label: 'last TTS (ms)', value: statsData.tts_ms },
])

const statusColor = computed(() => ({
  idle: 'grey', listening: 'success', speech: 'warning', tts: 'purple',
}[status.value]))
const statusLabel = computed(() => status.value.toUpperCase())

const peakPct = computed(() => Math.min(100, (currentPeak.value / MAX_PEAK) * 100))
const startThreshPct = computed(() => Math.min(100, ((config.vad_start_threshold || 20) / MAX_PEAK) * 100))
const stopThreshPct = computed(() => Math.min(100, ((config.vad_stop_threshold || 12) / MAX_PEAK) * 100))

// --- Log parsing ---
function parseTrace(line: string) {
  const clean = line.replace(/\x1b\[[0-9;]*m/g, '')
  const stageMatch = clean.match(/stage=(\S+)/)
  if (!stageMatch) return null
  let fields: Record<string, any> = {}
  const idx = clean.indexOf('fields={')
  if (idx !== -1) {
    const jsonStr = clean.substring(idx + 7)
    let depth = 0, end = -1
    for (let i = 0; i < jsonStr.length; i++) {
      if (jsonStr[i] === '{') depth++
      else if (jsonStr[i] === '}') { depth--; if (depth === 0) { end = i + 1; break } }
    }
    if (end > 0) {
      try { fields = JSON.parse(jsonStr.substring(0, end)) } catch {}
    }
  }
  return { stage: stageMatch[1], fields }
}

function addLog(text: string, level = 'info') {
  const time = new Date().toLocaleTimeString()
  logEntries.value.push({ time, text, level })
  if (logEntries.value.length > 200) logEntries.value.splice(0, logEntries.value.length - 200)
  nextTick(() => { if (logArea.value) logArea.value.scrollTop = logArea.value.scrollHeight })
}

function addWavePoint(peak: number, voiced: boolean) {
  waveData.value.push({ peak, voiced })
  if (waveData.value.length > MAX_WAVE * 2) waveData.value = waveData.value.slice(-MAX_WAVE)
  currentPeak.value = peak
  statsData.peak = peak
}

function processLine(line: string) {
  if (!line.includes('napcat_voice trace=')) return
  const parsed = parseTrace(line)
  if (!parsed) return
  const { stage, fields } = parsed

  switch (stage) {
    case 'turn.start':
      status.value = 'listening'
      addLog(`Turn started (start=${fields.vad_start_threshold}, stop=${fields.vad_stop_threshold})`)
      break
    case 'vad.speech_start':
      status.value = 'speech'
      addWavePoint(fields.peak || 0, true)
      addLog(`Speech detected! peak=${fields.peak}, threshold=${fields.start_threshold}`, 'ok')
      break
    case 'turn.vad.ok': {
      const v = fields.voiced_frames || 0, s = fields.silence_frames || 0
      const lastPeak = waveData.value.length > 0 ? waveData.value[waveData.value.length - 1].peak : 300
      for (let i = 0; i < Math.min(v, 40); i++) addWavePoint(Math.round(lastPeak * (0.5 + Math.random() * 0.8)), true)
      for (let i = 0; i < Math.min(s, 10); i++) addWavePoint(Math.round(3 + Math.random() * 8), false)
      statsData.voiced = v; statsData.silence = s
      addLog(`VAD OK: ${v} voiced, ${s} silence, ${fields.frame_count} total, ${fields.listen_elapsed_ms}ms`, 'ok')
      break
    }
    case 'turn.vad.failed': {
      status.value = 'listening'
      const rv = fields.recv || {}
      for (let i = 0; i < 3; i++) addWavePoint(Math.round(2 + Math.random() * 6), false)
      addLog(`VAD failed: ${fields.reason} (voiced=${rv.voiced_frames || 0})`, 'warn')
      break
    }
    case 'stt.ok': case 'turn.stt.ok':
      addLog(`STT: "${fields.recognized_text}" (${fields.attempt ? 'attempt ' + fields.attempt : ''})`, 'ok')
      break
    case 'stt.error': case 'turn.stt.failed': case 'turn.stt.filtered':
      addLog(`STT failed: ${fields.reason || fields.error || ''}`, 'warn')
      break
    case 'core.reply.done': case 'turn.reply.generated':
      addLog(`Reply: "${(fields.reply_text || '').substring(0, 80)}"`, 'info')
      break
    case 'turn.playback':
      status.value = 'tts'
      addLog(`TTS ${fields.ok !== false ? 'OK' : 'FAIL'}`, fields.ok !== false ? 'ok' : 'warn')
      break
    case 'loop.turn.end':
      statsData.turns = Math.max(statsData.turns, fields.turn_index || 0)
      status.value = fields.ok ? 'listening' : status.value
      addLog(`Turn ${fields.turn_index || '?'} ${fields.ok ? 'done' : 'failed: ' + (fields.reason || '')}`,
        fields.ok ? 'ok' : 'warn')
      break
  }
}

// --- Canvas ---
let animFrame = 0
function drawWaveform() {
  const canvas = waveCanvas.value
  if (!canvas) { animFrame = requestAnimationFrame(drawWaveform); return }
  const dpr = window.devicePixelRatio || 1
  if (canvas.width !== canvas.clientWidth * dpr || canvas.height !== canvas.clientHeight * dpr) {
    canvas.width = canvas.clientWidth * dpr
    canvas.height = canvas.clientHeight * dpr
  }
  const ctx = canvas.getContext('2d')!
  const w = canvas.width, h = canvas.height
  ctx.clearRect(0, 0, w, h)

  const startTh = config.vad_start_threshold || 20
  const stopTh = config.vad_stop_threshold || 12

  // Threshold lines
  const startY = h - (startTh / MAX_PEAK) * h
  const stopY = h - (stopTh / MAX_PEAK) * h
  ctx.setLineDash([4, 4])
  ctx.strokeStyle = '#ef5350'; ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(0, startY); ctx.lineTo(w, startY); ctx.stroke()
  ctx.strokeStyle = '#ffa726'
  ctx.beginPath(); ctx.moveTo(0, stopY); ctx.lineTo(w, stopY); ctx.stroke()
  ctx.setLineDash([])

  ctx.fillStyle = '#ef5350'; ctx.font = `${10 * dpr}px monospace`
  ctx.fillText(`start=${startTh}`, 4, startY - 3)
  ctx.fillStyle = '#ffa726'
  ctx.fillText(`stop=${stopTh}`, 4, stopY - 3)

  const data = waveData.value
  if (data.length < 2) { animFrame = requestAnimationFrame(drawWaveform); return }

  const step = w / MAX_WAVE
  const startIdx = Math.max(0, data.length - MAX_WAVE)

  // Line
  ctx.beginPath()
  ctx.strokeStyle = '#26c6da'; ctx.lineWidth = 2
  for (let i = startIdx; i < data.length; i++) {
    const x = (i - startIdx) * step
    const y = h - (Math.min(data[i].peak, MAX_PEAK) / MAX_PEAK) * h
    i === startIdx ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Dots
  for (let i = startIdx; i < data.length; i++) {
    const x = (i - startIdx) * step
    const y = h - (Math.min(data[i].peak, MAX_PEAK) / MAX_PEAK) * h
    ctx.fillStyle = data[i].voiced ? '#66bb6a' : '#555'
    ctx.beginPath(); ctx.arc(x, y, 2 * dpr, 0, Math.PI * 2); ctx.fill()
  }

  animFrame = requestAnimationFrame(drawWaveform)
}

// --- API ---
let pollTimer: ReturnType<typeof setInterval>
let logOffset = 0

async function pollLogs() {
  try {
    const { data } = await axios.get('/api/vad/log', { params: { offset: logOffset } })
    const resp = data.data || data
    if (resp.lines?.length) {
      resp.lines.forEach(processLine)
      logOffset = resp.new_offset ?? logOffset
    }
  } catch {}
}

async function loadConfig() {
  try {
    const { data } = await axios.get('/api/vad/config')
    const cfg = data.data || data
    Object.assign(config, cfg)
  } catch {}
}

async function saveConfig() {
  saving.value = true
  try {
    await axios.post('/api/vad/config', { ...config })
    addLog('Config saved! Use /vad_config reload in QQ or restart call to apply.', 'ok')
  } catch (e: any) {
    addLog('Config save failed: ' + (e.message || e), 'warn')
  } finally {
    saving.value = false
  }
}

function clearAll() {
  waveData.value = []
  logEntries.value = []
  statsData.voiced = 0; statsData.silence = 0; statsData.peak = 0
  statsData.turns = 0; statsData.stt_ms = '-'; statsData.tts_ms = '-'
  currentPeak.value = 0
  status.value = 'idle'
}

// --- Lifecycle ---
onMounted(() => {
  loadConfig()
  pollTimer = setInterval(pollLogs, POLL_MS)
  animFrame = requestAnimationFrame(drawWaveform)
  addLog('VAD Monitor ready. Start a voice call to see activity.')
})

onUnmounted(() => {
  clearInterval(pollTimer)
  cancelAnimationFrame(animFrame)
})
</script>

<style scoped>
.waveform-canvas {
  width: 100%;
  height: 160px;
  display: block;
  border-radius: 4px;
  background: rgb(var(--v-theme-surface));
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}
.peak-bar {
  height: 22px;
  background: rgb(var(--v-theme-surface));
  border-radius: 4px;
  overflow: hidden;
  position: relative;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}
.peak-fill {
  height: 100%;
  transition: width 0.1s;
  border-radius: 4px;
  background: linear-gradient(90deg, #66bb6a, #ffee58, #ef5350);
}
.threshold-line {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  z-index: 1;
}
.start-line { background: #ef5350; }
.stop-line { background: #ffa726; }
.peak-label {
  position: absolute;
  right: 8px;
  top: 2px;
  font-size: 12px;
  font-weight: bold;
  color: rgb(var(--v-theme-on-surface));
}
.stat-box {
  background: rgba(var(--v-theme-surface-variant), 0.3);
}
.log-area {
  height: 240px;
  overflow-y: auto;
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
  font-size: 12px;
  line-height: 1.7;
  background: rgb(var(--v-theme-surface));
  border-radius: 4px;
  padding: 8px;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}
.log-line { white-space: pre-wrap; }
.log-ok { color: #66bb6a; }
.log-warn { color: #ffa726; }
.log-err { color: #ef5350; }
.log-info { color: rgb(var(--v-theme-on-surface)); opacity: 0.8; }
</style>

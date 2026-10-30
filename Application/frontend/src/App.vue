<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import RiskChart from './components/RiskChart.vue'

const fallbackState = {
  system: {
    status: 'idle',
    message: '正在连接服务',
    engine_mode: null,
    engine_label: '尚未加载',
    device: '—',
    model: 'EdgeFall-ST HN3-B-R',
    fallback_reason: null,
  },
  stream: {
    clip_index: 0,
    clip_total: 18,
    clip_name: '客厅跌倒',
    scene: '室内 · 正面视角',
    dataset: 'OmniFall',
    expected: 'fall',
    elapsed_s: 0,
    duration_s: 0,
    progress: 0,
    loop_count: 0,
    playback_speed: 1,
    target_fps: 10,
  },
  inference: {
    score: 0,
    threshold: 0.7094,
    risk_level: 'normal',
    risk_label: '监测正常',
    geometry_score: 0,
    global_attention: null,
    roi_attention: null,
    actor_track_id: null,
    people_count: 0,
    latency_ms: 0,
    processing_fps: 0,
    breakdown: { detector_ms: 0, tracking_ms: 0, feature_pool_ms: 0, confirmer_ms: 0 },
  },
  active_event: null,
  score_history: [],
}

const telemetry = ref(structuredClone(fallbackState))
const config = ref({ clips: [], parameters: 12212714, version: 'v3.6-md-gcva-hn3br-final' })
const events = ref([])
const connected = ref(false)
const busy = ref(false)
const settingsOpen = ref(false)
const clipsOpen = ref(false)
const clipList = ref(null)
const soundEnabled = ref(true)
const toast = ref('')
const settings = ref({ threshold: 0.7094, playback_speed: 1 })
let socket
let reconnectTimer
let toastTimer
let audioContext
let lastAlertId = null

const risk = computed(() => telemetry.value.inference.risk_level)
const statusClass = computed(() => `is-${telemetry.value.system.status}`)
const isRunning = computed(() => telemetry.value.system.status === 'running')
const isPaused = computed(() => telemetry.value.system.status === 'paused')
const progressStyle = computed(() => ({ width: `${Math.min(100, telemetry.value.stream.progress * 100)}%` }))
const streamUrl = computed(() => `/api/stream/mjpeg?v=1`)
const latestEvents = computed(() => events.value.slice(0, 5))

function formatNumber(value, digits = 1) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : '—'
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0)
  const minutes = Math.floor(value / 60)
  const rest = Math.floor(value % 60)
  return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`
}

function eventTime(value) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).format(new Date(value))
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}

function scrollActiveClip() {
  if (!clipsOpen.value) return
  nextTick(() => {
    clipList.value?.querySelector('.active')?.scrollIntoView({ block: 'nearest' })
  })
}

function toggleClips() {
  clipsOpen.value = !clipsOpen.value
  if (clipsOpen.value) {
    settingsOpen.value = false
    scrollActiveClip()
  }
}

function openSettings() {
  clipsOpen.value = false
  settingsOpen.value = true
}

function handleKeydown(event) {
  if (event.key !== 'Escape') return
  clipsOpen.value = false
  settingsOpen.value = false
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || '请求失败')
  return response.json()
}

async function loadInitial() {
  try {
    const [stateData, configData, eventData] = await Promise.all([
      api('/api/state'), api('/api/config'), api('/api/events'),
    ])
    telemetry.value = stateData
    config.value = configData
    events.value = eventData.items
    settings.value.threshold = stateData.inference.threshold
    settings.value.playback_speed = stateData.stream.playback_speed
  } catch (error) {
    showToast(`后端连接失败：${error.message}`)
  }
}

function connectSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  socket = new WebSocket(`${protocol}//${location.host}/ws/telemetry`)
  socket.onopen = () => { connected.value = true }
  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data)
    telemetry.value = payload.type === 'heartbeat' ? payload.state : payload
  }
  socket.onerror = () => socket.close()
  socket.onclose = () => {
    connected.value = false
    reconnectTimer = setTimeout(connectSocket, 1800)
  }
}

async function control(action) {
  busy.value = true
  try {
    if (action === 'start' || action === 'resume') unlockAudio()
    telemetry.value = await api('/api/stream/control', {
      method: 'POST', body: JSON.stringify({ action }),
    })
    showToast({ start: '演示流正在启动', pause: '视频流已暂停', resume: '继续播放', restart: '已从第一段重新开始' }[action])
  } catch (error) {
    showToast(error.message)
  } finally {
    busy.value = false
  }
}

async function applySettings() {
  busy.value = true
  try {
    telemetry.value = await api('/api/settings', {
      method: 'PATCH', body: JSON.stringify(settings.value),
    })
    settingsOpen.value = false
    showToast('参数已应用到当前推理流')
  } catch (error) {
    showToast(error.message)
  } finally {
    busy.value = false
  }
}

async function refreshEvents() {
  try {
    events.value = (await api('/api/events')).items
  } catch {}
}

async function acknowledge(event) {
  try {
    await api(`/api/events/${event.id}/acknowledge`, { method: 'POST' })
    await refreshEvents()
    showToast('告警已确认并记录')
  } catch (error) {
    showToast(error.message)
  }
}

function unlockAudio() {
  if (!audioContext) audioContext = new AudioContext()
  if (audioContext.state === 'suspended') audioContext.resume()
}

function beep() {
  if (!soundEnabled.value) return
  unlockAudio()
  const oscillator = audioContext.createOscillator()
  const gain = audioContext.createGain()
  oscillator.type = 'sine'
  oscillator.frequency.setValueAtTime(880, audioContext.currentTime)
  gain.gain.setValueAtTime(0.0001, audioContext.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.12, audioContext.currentTime + 0.02)
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.35)
  oscillator.connect(gain).connect(audioContext.destination)
  oscillator.start()
  oscillator.stop(audioContext.currentTime + 0.36)
}

watch(() => telemetry.value.active_event?.id, async (id) => {
  if (id && id !== lastAlertId) {
    lastAlertId = id
    beep()
    await refreshEvents()
  }
})

watch(() => telemetry.value.stream.clip_index, scrollActiveClip)

onMounted(() => {
  loadInitial()
  connectSocket()
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  clearTimeout(reconnectTimer)
  clearTimeout(toastTimer)
  socket?.close()
  audioContext?.close()
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div class="app-shell" :class="`risk-${risk}`">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></div>
        <div>
          <p class="eyebrow">EDGEFALL-ST</p>
          <h1>实时跌倒检测</h1>
        </div>
      </div>
      <div class="header-actions">
        <div class="connection" :class="statusClass">
          <i></i>
          <span>{{ telemetry.system.message }}</span>
          <small>{{ connected ? '数据已连接' : '正在重连' }}</small>
        </div>
        <button class="icon-button" :aria-label="soundEnabled ? '关闭告警音' : '开启告警音'" @click="soundEnabled = !soundEnabled">
          {{ soundEnabled ? '声音开' : '声音关' }}
        </button>
        <button class="secondary-button" @click="openSettings">推理参数</button>
        <button
          class="secondary-button clips-trigger"
          :aria-expanded="clipsOpen"
          aria-controls="clips-drawer"
          @click="toggleClips"
        >
          <span>演示片段</span>
          <small>{{ telemetry.stream.clip_index + 1 }}/{{ telemetry.stream.clip_total }}</small>
        </button>
      </div>
    </header>

    <main>
      <section v-if="telemetry.active_event" class="active-alert" role="alert">
        <div class="alert-symbol">!</div>
        <div>
          <strong>检测到疑似跌倒，请及时查看</strong>
          <span>{{ telemetry.active_event.clip_name }} · 人员 ID {{ telemetry.active_event.track_id ?? '—' }} · 风险 {{ formatNumber(telemetry.active_event.score * 100, 1) }}%</span>
        </div>
        <button @click="acknowledge(telemetry.active_event)">确认已查看</button>
      </section>

      <section class="hero-grid">
        <article class="stream-panel">
          <div class="panel-heading stream-heading">
            <div>
              <div class="live-title">
                <span class="live-dot" :class="{ pulsing: isRunning }"></span>
                <h2>模拟摄像头 · CAM-01</h2>
              </div>
              <p>{{ telemetry.stream.scene }} · {{ telemetry.stream.dataset }}</p>
            </div>
            <div class="stream-meta">
              <span>{{ telemetry.stream.clip_index + 1 }} / {{ telemetry.stream.clip_total }}</span>
              <b>{{ telemetry.stream.clip_name }}</b>
            </div>
          </div>

          <div class="video-stage">
            <img :src="streamUrl" alt="实时跌倒检测视频流" />
            <div v-if="telemetry.system.status === 'loading'" class="video-loading">
              <span class="loader"></span>
              <strong>正在加载推理引擎</strong>
              <small>首次加载模型可能需要数秒</small>
            </div>
            <div class="video-badge">{{ telemetry.system.engine_label }}</div>
            <div class="video-time">{{ formatTime(telemetry.stream.elapsed_s) }} / {{ formatTime(telemetry.stream.duration_s) }}</div>
          </div>
          <div class="clip-progress"><span :style="progressStyle"></span></div>

          <div class="stream-controls">
            <button v-if="!isRunning" class="primary-button" :disabled="busy || telemetry.system.status === 'loading'" @click="control(isPaused ? 'resume' : 'start')">
              <span class="play-symbol">▶</span>{{ isPaused ? '继续监测' : '开始演示' }}
            </button>
            <button v-else class="primary-button pause" :disabled="busy" @click="control('pause')"><span>Ⅱ</span>暂停</button>
            <button class="text-button" :disabled="busy" @click="control('restart')">从头播放</button>
            <div class="control-spacer"></div>
            <span class="loop-label">第 {{ telemetry.stream.loop_count + 1 }} 轮 · {{ telemetry.stream.playback_speed }}×</span>
          </div>
        </article>

        <aside class="risk-panel" :class="`risk-card-${risk}`">
          <div class="panel-heading">
            <div>
              <p class="section-kicker">当前判定</p>
              <h2>{{ telemetry.inference.risk_label }}</h2>
            </div>
            <span class="risk-chip">{{ risk === 'danger' ? '高风险' : risk === 'warning' ? '关注' : '正常' }}</span>
          </div>

          <div class="score-wrap">
            <div class="score-ring" :style="{ '--score': telemetry.inference.score * 360 + 'deg' }">
              <div>
                <strong>{{ formatNumber(telemetry.inference.score * 100, 1) }}</strong>
                <span>%</span>
              </div>
            </div>
            <p>跌倒风险分数</p>
          </div>

          <div class="threshold-block">
            <div><span>报警阈值</span><b>{{ formatNumber(telemetry.inference.threshold * 100, 1) }}%</b></div>
            <div class="threshold-track">
              <span class="score-fill" :style="{ width: `${telemetry.inference.score * 100}%` }"></span>
              <i :style="{ left: `${telemetry.inference.threshold * 100}%` }"></i>
            </div>
          </div>

          <dl class="risk-details">
            <div><dt>画面人数</dt><dd>{{ telemetry.inference.people_count }} 人</dd></div>
            <div><dt>重点人员</dt><dd>{{ telemetry.inference.actor_track_id == null ? '未锁定' : `ID ${telemetry.inference.actor_track_id}` }}</dd></div>
            <div><dt>姿态分数</dt><dd>{{ formatNumber(telemetry.inference.geometry_score, 3) }}</dd></div>
            <div><dt>单帧耗时</dt><dd>{{ formatNumber(telemetry.inference.latency_ms, 1) }} ms</dd></div>
          </dl>
        </aside>
      </section>

      <section class="metric-strip" aria-label="运行指标">
        <div><span>推理速度</span><strong>{{ formatNumber(telemetry.inference.processing_fps, 1) }}</strong><small>FPS</small></div>
        <div><span>检测耗时</span><strong>{{ formatNumber(telemetry.inference.breakdown.detector_ms, 1) }}</strong><small>ms</small></div>
        <div><span>跟踪耗时</span><strong>{{ formatNumber(telemetry.inference.breakdown.tracking_ms, 1) }}</strong><small>ms</small></div>
        <div><span>确认器耗时</span><strong>{{ formatNumber(telemetry.inference.breakdown.confirmer_ms, 1) }}</strong><small>ms</small></div>
        <div class="device-metric"><span>运行设备</span><strong>{{ telemetry.system.device }}</strong><small>{{ telemetry.system.engine_mode === 'live' ? '在线计算' : '演示数据' }}</small></div>
      </section>

      <section class="lower-grid">
        <article class="trend-panel">
          <div class="panel-heading">
            <div>
              <p class="section-kicker">最近趋势</p>
              <h2>跌倒风险变化</h2>
            </div>
            <div class="legend"><span class="line-current"></span>实时分数 <span class="line-threshold"></span>报警阈值</div>
          </div>
          <RiskChart :history="telemetry.score_history" :threshold="telemetry.inference.threshold" />
        </article>

        <article class="events-panel">
          <div class="panel-heading">
            <div>
              <p class="section-kicker">事件记录</p>
              <h2>最近告警</h2>
            </div>
            <button class="text-button compact" @click="refreshEvents">刷新</button>
          </div>
          <div v-if="!latestEvents.length" class="empty-events">
            <span>✓</span>
            <div><strong>暂无跌倒告警</strong><p>检测到高风险事件后会显示在这里</p></div>
          </div>
          <ul v-else class="event-list">
            <li v-for="event in latestEvents" :key="event.id">
              <span class="event-dot" :class="event.status"></span>
              <div><strong>{{ event.clip_name }}</strong><small>{{ eventTime(event.created_at) }} · ID {{ event.track_id ?? '—' }}</small></div>
              <b>{{ formatNumber(event.score * 100, 1) }}%</b>
              <button v-if="event.status === 'pending'" @click="acknowledge(event)">确认</button>
              <em v-else>已处理</em>
            </li>
          </ul>
        </article>
      </section>

    </main>

    <footer>
      <span>EdgeFall-ST 在线演示</span>
      <span>模型 {{ config.version }} · {{ (config.parameters / 1_000_000).toFixed(1) }}M 参数</span>
      <span>演示输出仅用于模型能力展示，不替代人工处置</span>
    </footer>

    <div v-if="settingsOpen" class="drawer-backdrop" @click.self="settingsOpen = false">
      <aside class="settings-drawer" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <div class="drawer-header">
          <div><p class="section-kicker">当前视频流</p><h2 id="settings-title">推理参数</h2></div>
          <button class="close-button" aria-label="关闭" @click="settingsOpen = false">×</button>
        </div>

        <label class="setting-field">
          <span><b>报警阈值</b><output>{{ formatNumber(settings.threshold * 100, 1) }}%</output></span>
          <input v-model.number="settings.threshold" type="range" min="0.1" max="0.95" step="0.01" />
          <small>分数达到该阈值时触发跌倒告警。默认验证集最优值为 70.94%。</small>
        </label>

        <label class="setting-field">
          <span><b>播放速度</b><output>{{ settings.playback_speed }}×</output></span>
          <select v-model.number="settings.playback_speed">
            <option :value="0.5">0.5× 慢速查看</option>
            <option :value="1">1× 实时速度</option>
            <option :value="1.5">1.5× 快速演示</option>
            <option :value="2">2× 快速演示</option>
          </select>
          <small>仅调整模拟推流的播放节奏，不改变模型采样频率。</small>
        </label>

        <div class="engine-info">
          <span>推理引擎</span>
          <strong>{{ telemetry.system.engine_label }}</strong>
          <p>{{ telemetry.system.engine_mode === 'live' ? '模型正在对每个采样帧在线计算。' : '使用已保存的真实推理结果同步回放；配置 EDGEFALL_ENGINE=live 可强制启用在线模型。' }}</p>
        </div>

        <dl class="fixed-params">
          <div><dt>模型采样</dt><dd>{{ config.runtime?.sample_fps ?? 10 }} FPS</dd></div>
          <div><dt>人员置信度</dt><dd>{{ config.runtime?.detector_confidence ?? 0.1 }}</dd></div>
          <div><dt>NMS IoU</dt><dd>{{ config.runtime?.nms_iou ?? 0.3 }}</dd></div>
          <div><dt>最多人员</dt><dd>{{ config.runtime?.max_people ?? 2 }} 人</dd></div>
        </dl>

        <div v-if="telemetry.system.fallback_reason" class="fallback-note">
          当前环境未能加载实时模型，已安全切换为回放模式。
        </div>

        <button class="primary-button full" :disabled="busy" @click="applySettings">应用参数</button>
      </aside>
    </div>

    <Transition name="drawer-fade">
      <div v-if="clipsOpen" class="drawer-backdrop clips-backdrop" @click.self="clipsOpen = false">
        <aside
          id="clips-drawer"
          class="clips-drawer"
          role="dialog"
          aria-modal="true"
          aria-labelledby="clips-title"
        >
          <div class="drawer-header clips-drawer-header">
            <div>
              <p class="section-kicker">演示片段</p>
              <h2 id="clips-title">循环推流序列</h2>
            </div>
            <button class="close-button" aria-label="隐藏演示片段" @click="clipsOpen = false">×</button>
          </div>

          <div class="active-clip-summary">
            <span>正在播放 {{ telemetry.stream.clip_index + 1 }} / {{ telemetry.stream.clip_total }}</span>
            <strong>{{ telemetry.stream.clip_name }}</strong>
            <small>{{ telemetry.stream.dataset }} · 第 {{ telemetry.stream.loop_count + 1 }} 轮</small>
          </div>

          <ol ref="clipList" class="clip-drawer-list">
            <li
              v-for="clip in config.clips"
              :key="clip.index"
              :class="{ active: clip.index === telemetry.stream.clip_index }"
            >
              <span>{{ String(clip.index + 1).padStart(2, '0') }}</span>
              <div>
                <strong>{{ clip.title }}</strong>
                <small>{{ clip.scene }} · {{ clip.dataset }}</small>
              </div>
              <i>{{ clip.expected === 'fall' ? '跌倒' : '日常' }}</i>
            </li>
          </ol>

          <button class="hide-clips-button" @click="clipsOpen = false">隐藏片段列表</button>
        </aside>
      </div>
    </Transition>

    <Transition name="toast"><div v-if="toast" class="toast">{{ toast }}</div></Transition>
  </div>
</template>

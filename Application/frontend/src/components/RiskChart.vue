<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  history: { type: Array, default: () => [] },
  threshold: { type: Number, default: 0.7094 },
})

const canvas = ref(null)
let observer

function render() {
  const element = canvas.value
  if (!element) return
  const rect = element.getBoundingClientRect()
  const ratio = window.devicePixelRatio || 1
  element.width = Math.max(1, Math.round(rect.width * ratio))
  element.height = Math.max(1, Math.round(rect.height * ratio))
  const ctx = element.getContext('2d')
  ctx.scale(ratio, ratio)

  const width = rect.width
  const height = rect.height
  const top = 12
  const bottom = 24
  const left = 4
  const right = 38
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  ctx.clearRect(0, 0, width, height)

  ctx.strokeStyle = '#dce4e0'
  ctx.lineWidth = 1
  ;[0, 0.5, 1].forEach((value) => {
    const y = top + (1 - value) * plotHeight
    ctx.beginPath()
    ctx.moveTo(left, y)
    ctx.lineTo(width - right, y)
    ctx.stroke()
    ctx.fillStyle = '#82908c'
    ctx.font = '11px Inter, system-ui, sans-serif'
    ctx.fillText(value.toFixed(1), width - right + 8, y + 4)
  })

  const thresholdY = top + (1 - props.threshold) * plotHeight
  ctx.setLineDash([5, 5])
  ctx.strokeStyle = '#d96255'
  ctx.beginPath()
  ctx.moveTo(left, thresholdY)
  ctx.lineTo(width - right, thresholdY)
  ctx.stroke()
  ctx.setLineDash([])

  if (!props.history.length) return
  const points = props.history.slice(-120)
  ctx.beginPath()
  points.forEach((point, index) => {
    const x = left + (index / Math.max(1, points.length - 1)) * plotWidth
    const y = top + (1 - Math.max(0, Math.min(1, point.score))) * plotHeight
    if (index === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.strokeStyle = '#327d69'
  ctx.lineWidth = 2.5
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.stroke()

  const last = points.at(-1)
  const lastX = left + plotWidth
  const lastY = top + (1 - Math.max(0, Math.min(1, last.score))) * plotHeight
  ctx.fillStyle = last.score >= props.threshold ? '#d94f43' : '#327d69'
  ctx.beginPath()
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2)
  ctx.fill()
}

watch(() => [props.history, props.threshold], render, { deep: true })
onMounted(() => {
  observer = new ResizeObserver(render)
  observer.observe(canvas.value)
  render()
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <canvas ref="canvas" class="risk-chart" aria-label="最近跌倒风险分数趋势图"></canvas>
</template>

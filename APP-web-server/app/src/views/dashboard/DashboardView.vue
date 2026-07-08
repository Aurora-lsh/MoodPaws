<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Capacitor } from "@capacitor/core";
import { getVideoStreamUrl } from "../../config/api";
import { usePetApi } from "../../composables/usePetApi";

const petPhotoUrl =
  "https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=240&q=80";

const emptyHouseState = {
  name: "当前宠舍",
  petName: "Lucky",
  petProfile: "金毛寻回犬 · 3 岁",
  notificationCount: 0,
  environment: {
    temperature: { value: "--", status: "等待数据", unit: "°C" },
    humidity: { value: "--", unit: "%" },
    airQuality: { value: "--", unit: "" },
    co2: { value: "--", unit: "ppm" },
  },
  liveView: {
    hasVideo: true,
    status: "在线看护",
    videoStreamUrl: "",
  },
  emotion: {
    primary: "开心",
    secondary: "想玩耍",
  },
};

const {
  loading,
  errorMessage,
  latestTelemetry,
  metricSections,
  metricMap,
  emotion,
  refreshTelemetryBundle,
  refreshEmotionBundle,
  startTelemetryPolling,
  startEmotionPolling,
  stopTelemetryPolling,
  stopEmotionPolling,
} = usePetApi();

const hasVideoError = ref(false);
const isVideoConnecting = ref(false);
const hasVideoLoaded = ref(false);
const videoViewport = ref(null);
const videoFrameScale = ref(1);
const reconnectAttempt = ref(0);
let deferredVideoDiscoveryTask = null;
let videoReadyFallbackTimer = null;
let videoFrameResizeObserver = null;

const VIDEO_FRAME_WIDTH = 640;
const VIDEO_FRAME_HEIGHT = 480;

const getMetric = (keys) =>
  computed(() => {
    const keyList = Array.isArray(keys) ? keys : [keys];
    return keyList.map((key) => metricMap.value[key]).find((metric) => metric);
  });

const temperatureMetric = getMetric(["PetHouse:Temp"]);
const humidityMetric = getMetric(["PetHouse:Humi"]);
const co2Metric = getMetric(["PetHouse:CO2"]);
const airQualityMetric = getMetric(["PetHouse:MQ135"]);
const resolvedVideoStreamUrl = computed(() => getVideoStreamUrl());
const hasTelemetryData = computed(() => Boolean(metricSections.value.length));

function scheduleDeferredTask(callback, timeout = 240) {
  if (typeof window === "undefined") {
    return null;
  }

  if (typeof window.requestIdleCallback === "function") {
    return window.requestIdleCallback(callback, { timeout });
  }

  return window.setTimeout(callback, timeout);
}

function clearDeferredTask(taskId) {
  if (taskId == null || typeof window === "undefined") {
    return;
  }

  if (typeof window.cancelIdleCallback === "function") {
    window.cancelIdleCallback(taskId);
    return;
  }

  window.clearTimeout(taskId);
}

function formatMetricDisplay(metric, fallback) {
  const value = metric?.value;
  const unit = metric?.unit || fallback.unit || "";

  if (value === undefined || value === null || value === "") {
    return {
      value: fallback.value,
      unit,
    };
  }

  return {
    value,
    unit,
  };
}

const petProfile = computed(() => {
  const sourceDevice = latestTelemetry.value.source?.deviceName;
  return sourceDevice && sourceDevice !== "--"
    ? "设备已连接"
    : emptyHouseState.petProfile;
});

const petName = computed(() => {
  return emptyHouseState.petName;
});

const selectedHouse = computed(() => ({
  ...emptyHouseState,
  petName: petName.value,
  petProfile: petProfile.value,
  liveView: {
    ...emptyHouseState.liveView,
    hasVideo: Boolean(resolvedVideoStreamUrl.value),
    videoStreamUrl: resolvedVideoStreamUrl.value,
  },
  environment: {
    temperature: {
      value:
        temperatureMetric.value?.value ??
        emptyHouseState.environment.temperature.value,
      status: temperatureMetric.value
        ? "实时同步"
        : emptyHouseState.environment.temperature.status,
      unit:
        temperatureMetric.value?.unit ||
        emptyHouseState.environment.temperature.unit,
    },
    humidity: {
      ...formatMetricDisplay(
        humidityMetric.value,
        emptyHouseState.environment.humidity,
      ),
    },
    airQuality: formatMetricDisplay(
      airQualityMetric.value,
      emptyHouseState.environment.airQuality,
    ),
    co2: formatMetricDisplay(co2Metric.value, emptyHouseState.environment.co2),
  },
  emotion: {
    primary: emotion.value?.currentMood || emptyHouseState.emotion.primary,
    secondary: errorMessage.value
      ? "稍后再试"
      : loading.value
        ? "正在同步"
        : emptyHouseState.emotion.secondary,
  },
}));

const streamSrc = computed(() => {
  const url = selectedHouse.value.liveView.videoStreamUrl;
  if (!url) return "";
  return reconnectAttempt.value === 0
    ? url
    : `${url}${url.includes("?") ? "&" : "?"}t=${reconnectAttempt.value}`;
});

const shouldUseNativeVideoFrame = computed(
  () => Capacitor.isNativePlatform() && Boolean(streamSrc.value),
);

const videoFrameStyle = computed(() => {
  if (!shouldUseNativeVideoFrame.value) {
    return null;
  }

  return {
    width: `${VIDEO_FRAME_WIDTH}px`,
    height: `${VIDEO_FRAME_HEIGHT}px`,
    transform: `translate(-50%, -50%) scale(${videoFrameScale.value})`,
  };
});

const liveViewStatus = computed(() => {
  if (!selectedHouse.value.liveView.hasVideo) {
    if (isVideoConnecting.value) return "搜索中";
    return loading.value ? "发现视频中" : "暂无画面";
  }

  if (isVideoConnecting.value) return "连接中";
  if (hasVideoError.value) return "暂时离线";
  return selectedHouse.value.liveView.status;
});

function handleStreamLoad() {
  clearVideoReadyFallback();
  hasVideoLoaded.value = true;
  isVideoConnecting.value = false;
  hasVideoError.value = false;
}

function handleStreamError() {
  clearVideoReadyFallback();
  hasVideoLoaded.value = false;
  isVideoConnecting.value = false;
  hasVideoError.value = true;
}

async function reconnectStream() {
  if (!selectedHouse.value.liveView.hasVideo) {
    openVideoSettings();
    return;
  }

  hasVideoError.value = false;
  hasVideoLoaded.value = false;
  isVideoConnecting.value = true;
  reconnectAttempt.value += 1;
  scheduleVideoReadyFallback();
}

function scheduleVideoDiscovery() {
  clearDeferredTask(deferredVideoDiscoveryTask);
  deferredVideoDiscoveryTask = null;
  hasVideoError.value = false;
  isVideoConnecting.value =
    Boolean(resolvedVideoStreamUrl.value) && !hasVideoLoaded.value;
  scheduleVideoReadyFallback();
}

function openVideoSettings() {
  window.dispatchEvent(new Event("moodpaws:open-api-sheet"));
}

function clearVideoReadyFallback() {
  if (videoReadyFallbackTimer == null || typeof window === "undefined") {
    return;
  }

  window.clearTimeout(videoReadyFallbackTimer);
  videoReadyFallbackTimer = null;
}

function scheduleVideoReadyFallback() {
  clearVideoReadyFallback();

  if (
    !streamSrc.value ||
    !isVideoConnecting.value ||
    typeof window === "undefined"
  ) {
    return;
  }

  videoReadyFallbackTimer = window.setTimeout(() => {
    if (!isVideoConnecting.value || hasVideoError.value || !streamSrc.value) {
      return;
    }

    hasVideoLoaded.value = true;
    isVideoConnecting.value = false;
  }, 1800);
}

function updateVideoFrameScale() {
  const viewport = videoViewport.value;
  if (!viewport) {
    return;
  }

  const rect = viewport.getBoundingClientRect();
  const widthScale = rect.width / VIDEO_FRAME_WIDTH;
  const heightScale = rect.height / VIDEO_FRAME_HEIGHT;
  videoFrameScale.value = Math.min(widthScale, heightScale, 1);
}

function startVideoFrameScaling() {
  updateVideoFrameScale();

  if (typeof ResizeObserver === "undefined" || !videoViewport.value) {
    window.addEventListener("resize", updateVideoFrameScale);
    return;
  }

  videoFrameResizeObserver = new ResizeObserver(updateVideoFrameScale);
  videoFrameResizeObserver.observe(videoViewport.value);
}

function stopVideoFrameScaling() {
  window.removeEventListener("resize", updateVideoFrameScale);

  if (videoFrameResizeObserver) {
    videoFrameResizeObserver.disconnect();
    videoFrameResizeObserver = null;
  }
}

watch(
  resolvedVideoStreamUrl,
  (url, previousUrl) => {
    if (!url) {
      hasVideoError.value = false;
      isVideoConnecting.value = false;
      hasVideoLoaded.value = false;
      reconnectAttempt.value = 0;
      clearVideoReadyFallback();
      return;
    }

    if (url !== previousUrl) {
      hasVideoError.value = false;
      hasVideoLoaded.value = false;
      isVideoConnecting.value = true;
      reconnectAttempt.value = 0;
      scheduleVideoReadyFallback();
    }
  },
  { immediate: true },
);

onMounted(async () => {
  await Promise.all([
    refreshTelemetryBundle({ includeHistory: false, includeTrack: false }),
    refreshEmotionBundle(),
  ]);
  startTelemetryPolling();
  startEmotionPolling();
  scheduleVideoDiscovery();
  await nextTick();
  startVideoFrameScaling();
});

onBeforeUnmount(() => {
  stopTelemetryPolling();
  stopEmotionPolling();
  clearDeferredTask(deferredVideoDiscoveryTask);
  clearVideoReadyFallback();
  stopVideoFrameScaling();
});
</script>

<template>
  <div class="pet-house-view">
    <section class="boarding-card">
      <header class="card-header">
        <div class="header-copy">
          <span class="page-kicker">宠舍总览</span>
          <h1 class="page-title">今天也在好好陪伴</h1>
        </div>

        <div class="header-top">
          <div class="dog-avatar-container">
            <img class="dog-avatar" :src="petPhotoUrl" alt="可爱的宠物照片" />
          </div>
          <div class="boarding-info">
            <div class="house-name">{{ selectedHouse.name }}</div>
            <div class="buddy-info">
              <span
                class="status-dot"
                :class="{ 'is-muted': !hasTelemetryData }"
              ></span
              >{{ selectedHouse.petName }} ·
              {{ selectedHouse.petProfile }}
            </div>
          </div>
          <div class="notification-bell" aria-label="照护提醒">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
              <path d="M13.73 21a2 2 0 0 1-3.46 0" />
            </svg>
            <span v-if="selectedHouse.notificationCount" class="badge">{{
              selectedHouse.notificationCount
            }}</span>
          </div>
        </div>
      </header>

      <section class="section">
        <div class="section-heading">
          <h3 class="section-title">环境监测</h3>
          <span class="section-note">{{
            selectedHouse.environment.temperature.status
          }}</span>
        </div>
        <div class="env-grid">
          <div class="env-card temperature">
            <div style="display: flex; gap: 20px">
              <div class="env-icon env-icon-temperature" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M12 4a2 2 0 0 0-2 2v7.2a4 4 0 1 0 4 0V6a2 2 0 0 0-2-2Z"
                    stroke="currentColor"
                    stroke-width="1.8"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <path
                    d="M12 10v6"
                    stroke="currentColor"
                    stroke-width="1.8"
                    stroke-linecap="round"
                  />
                </svg>
              </div>
              <div class="env-label">温度</div>
            </div>

            <div class="env-value">
              <span>{{ selectedHouse.environment.temperature.value }}</span>
              <span class="env-unit">{{
                selectedHouse.environment.temperature.unit
              }}</span>
            </div>
          </div>

          <div class="env-card humidity">
            <div style="display: flex; gap: 20px">
              <div class="env-icon env-icon-humidity" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M12 3.5s-5 6.1-5 10a5 5 0 0 0 10 0c0-3.9-5-10-5-10Z"
                    stroke="currentColor"
                    stroke-width="1.8"
                    stroke-linejoin="round"
                  />
                </svg>
              </div>
              <div class="env-label">湿度</div>
            </div>
            <div class="env-value">
              <span>{{ selectedHouse.environment.humidity.value }}</span>
              <span class="env-unit">{{
                selectedHouse.environment.humidity.unit
              }}</span>
            </div>
          </div>

          <div class="env-card air-quality air-quality-combined">
            <div class="env-metric-group">
              <div class="env-label">空气质量</div>
              <div class="env-value">
                <span>{{ selectedHouse.environment.airQuality.value }}</span>
                <span
                  v-if="selectedHouse.environment.airQuality.unit"
                  class="env-unit"
                  >{{ selectedHouse.environment.airQuality.unit }}</span
                >
              </div>
            </div>
            <div class="env-metric-group env-metric-group-secondary">
              <div class="env-sub-label">CO2 浓度</div>
              <div class="env-value env-value-secondary">
                <span>{{ selectedHouse.environment.co2.value }}</span>
                <span
                  v-if="selectedHouse.environment.co2.unit"
                  class="env-unit"
                  >{{ selectedHouse.environment.co2.unit }}</span
                >
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="section">
        <div class="section-heading">
          <h3 class="section-title">实时看护</h3>
        </div>
        <div class="live-view">
          <div ref="videoViewport" class="video-placeholder">
            <iframe
              v-if="
                selectedHouse.liveView.hasVideo &&
                !hasVideoError &&
                shouldUseNativeVideoFrame
              "
              :key="streamSrc"
              :src="streamSrc"
              title="宠舍监控画面"
              class="video-stream video-frame"
              :class="{ connecting: isVideoConnecting }"
              :style="videoFrameStyle"
              scrolling="no"
              @load="handleStreamLoad"
              @error="handleStreamError"
            ></iframe>
            <img
              v-else-if="selectedHouse.liveView.hasVideo && !hasVideoError"
             
              :src="streamSrc"
              alt="宠舍监控画面"
              class="video-stream"
              :class="{ connecting: isVideoConnecting }"
              @load="handleStreamLoad"
              @error="handleStreamError"
            />
            <div v-if="isVideoConnecting" class="video-overlay connecting">
              <span class="video-spinner" aria-hidden="true"></span>
              <span class="video-overlay-text"
                >正在连接看护画面，请稍等一下</span
              >
            </div>
            <div
              v-else-if="!selectedHouse.liveView.hasVideo || hasVideoError"
              class="video-overlay offline"
            >
              <span class="video-icon">看护</span>
              <span class="video-overlay-text">
                {{
                  selectedHouse.liveView.hasVideo
                    ? "看护画面暂时离线，宠舍数据仍会继续同步"
                    : "还没有填写视频地址，可在服务设置里补充 RDK 视频流地址"
                }}
              </span>
              <button
                type="button"
                class="reconnect-button"
                @click="reconnectStream"
              >
                {{
                  selectedHouse.liveView.hasVideo ? "重新连接" : "去填写地址"
                }}
              </button>
            </div>
            <span
              class="video-badge"
              :class="{ offline: hasVideoError, connecting: isVideoConnecting }"
              >{{ liveViewStatus }}</span
            >
          </div>
        </div>
      </section>
    </section>
  </div>
</template>

<style scoped src="./DashboardView.css"></style>

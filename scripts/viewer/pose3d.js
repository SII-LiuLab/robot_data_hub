/* Contract-only TCP pose display. Each arm is drawn in its own fixed reference frame. */
class Pose3D {
  constructor(canvas, stream, side) {
    this.canvas = canvas;
    this.stream = stream;
    this.side = side;
    this.yaw = -Math.PI / 2;
    this.pitch = 0.55;
    this.zoom = 1;
    this.index = -2;
    this.openness = null;
    this.drag = null;
    this.trailNs = 2e9;
    const initial = stream.values[0];
    const x = initial.slice(3, 6), y = initial.slice(6, 9);
    const z = [x[1] * y[2] - x[2] * y[1],
               x[2] * y[0] - x[0] * y[2],
               x[0] * y[1] - x[1] * y[0]];
    this.initialAxes = [x, y, z];
    const frame = this.getEpisodeFrame();
    this.center = frame.center;
    this.span = frame.span;
    this.onDown = event => {
      this.drag = [event.clientX, event.clientY];
      canvas.setPointerCapture(event.pointerId);
    };
    this.onMove = event => {
      if (!this.drag) return;
      this.yaw += (event.clientX - this.drag[0]) * 0.009;
      this.pitch = Math.max(-1.45, Math.min(1.45,
        this.pitch + (event.clientY - this.drag[1]) * 0.009));
      this.drag = [event.clientX, event.clientY];
      this.draw();
    };
    this.onUp = () => { this.drag = null; };
    this.onWheel = event => {
      event.preventDefault();
      this.zoom = Math.max(0.3, Math.min(5, this.zoom * Math.exp(-event.deltaY * 0.001)));
      this.draw();
    };
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    canvas.addEventListener('pointerup', this.onUp);
    canvas.addEventListener('pointercancel', this.onUp);
    canvas.addEventListener('wheel', this.onWheel, {passive: false});
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener('pointerdown', this.onDown);
    this.canvas.removeEventListener('pointermove', this.onMove);
    this.canvas.removeEventListener('pointerup', this.onUp);
    this.canvas.removeEventListener('pointercancel', this.onUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
  }

  reset() {
    this.yaw = -Math.PI / 2; this.pitch = 0.55; this.zoom = 1; this.draw();
  }

  trailStart(index) {
    const times = this.stream.times;
    const cutoff = times[index] - this.trailNs;
    let low = 0, high = index;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (times[middle] < cutoff) low = middle + 1; else high = middle;
    }
    return low;
  }

  getEpisodeFrame() {
    const origin = this.stream.values[0].slice(0, 3);
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (const pose of this.stream.values) {
      const offset = pose.slice(0, 3).map((value, i) => value - origin[i]);
      this.initialAxes.forEach((axis, i) => {
        const value = axis.reduce((sum, component, j) => sum + component * offset[j], 0);
        min[i] = Math.min(min[i], value);
        max[i] = Math.max(max[i], value);
      });
    }
    const middle = min.map((value, i) => (value + max[i]) / 2);
    return {center: this.pointFromInitialAxes(origin, middle),
            span: Math.max(0.25, ...min.map((value, i) => max[i] - value)) * 1.8};
  }

  setTrailDuration(durationNs) {
    this.trailNs = durationNs;
    this.draw();
  }

  setSample(index, openness) {
    if (this.index === index && this.openness === openness) return;
    this.index = index;
    this.openness = openness;
    this.draw();
  }

  project(point, width, height) {
    const offset = point.map((v, i) => v - this.center[i]);
    const p = this.initialAxes.map(axis => axis.reduce(
      (sum, value, i) => sum + value * offset[i], 0));
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const scale = Math.min(width, height) * 0.62 / this.span * this.zoom;
    const horizontal = cy * p[0] - sy * p[1];
    const depth = sy * p[0] + cy * p[1];
    return [width / 2 + horizontal * scale, height / 2 + (sp * depth - cp * p[2]) * scale];
  }

  pointFromInitialAxes(center, local) {
    return center.map((value, i) => value + this.initialAxes.reduce(
      (sum, axis, j) => sum + axis[i] * local[j], 0));
  }

  line(ctx, a, b, width, height, color, thickness = 1) {
    const start = this.project(a, width, height), end = this.project(b, width, height);
    ctx.strokeStyle = color; ctx.lineWidth = thickness;
    ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...end); ctx.stroke();
    return end;
  }

  arrow(ctx, origin, direction, length, width, height, color, label) {
    const tip = origin.map((v, i) => v + direction[i] * length);
    const start2d = this.project(origin, width, height);
    const end2d = this.line(ctx, origin, tip, width, height, color, 3);
    const angle = Math.atan2(end2d[1] - start2d[1], end2d[0] - start2d[0]);
    ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(...end2d);
    ctx.lineTo(end2d[0] - 9 * Math.cos(angle - 0.45), end2d[1] - 9 * Math.sin(angle - 0.45));
    ctx.lineTo(end2d[0] - 9 * Math.cos(angle + 0.45), end2d[1] - 9 * Math.sin(angle + 0.45));
    ctx.closePath(); ctx.fill();
    ctx.font = 'bold 13px system-ui'; ctx.fillText(label, end2d[0] + 5, end2d[1] - 5);
  }

  drawTrail(ctx, width, height, from, to) {
    const times = this.stream.times, values = this.stream.values;
    const step = Math.max(1, Math.ceil((to - from) / 250));
    const color = this.side === 'left' ? '116,199,255' : '255,189,127';
    const duration = Math.max(1, times[to] - times[from]);
    for (let i = from; i < to;) {
      const next = Math.min(to, i + step);
      const age = (times[next] - times[from]) / duration;
      this.line(ctx, values[i].slice(0, 3), values[next].slice(0, 3),
        width, height, `rgba(${color},${(0.14 + 0.82 * age).toFixed(3)})`, 1.5 + age * 1.5);
      i = next;
    }
  }

  draw() {
    const canvas = this.canvas, width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const dpi = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpi); canvas.height = Math.round(height * dpi);
    const ctx = canvas.getContext('2d'); ctx.scale(dpi, dpi);
    ctx.fillStyle = '#0c1520'; ctx.fillRect(0, 0, width, height);
    if (this.index < 0) {
      ctx.fillStyle = '#aabbd0'; ctx.font = '14px system-ui';
      ctx.fillText('等待首个 TCP 样本', 12, height / 2);
      return;
    }
    const from = this.trailStart(this.index);
    const center = this.center, span = this.span;
    for (let step = -2; step <= 2; step++) {
      const offset = step * span / 4;
      this.line(ctx,
        this.pointFromInitialAxes(center, [offset, -span / 2, -span * 0.45]),
        this.pointFromInitialAxes(center, [offset, span / 2, -span * 0.45]),
        width, height, '#213346');
      this.line(ctx,
        this.pointFromInitialAxes(center, [-span / 2, offset, -span * 0.45]),
        this.pointFromInitialAxes(center, [span / 2, offset, -span * 0.45]),
        width, height, '#213346');
    }
    this.drawTrail(ctx, width, height, from, this.index);
    if (this.index >= 0) {
      const pose = this.stream.values[this.index];
      const position = pose.slice(0, 3);
      const x = pose.slice(3, 6), y = pose.slice(6, 9);
      const z = [x[1] * y[2] - x[2] * y[1],
                 x[2] * y[0] - x[0] * y[2],
                 x[0] * y[1] - x[1] * y[0]];
      const axisLength = Math.max(0.055, Math.min(0.2, span * 0.18));
      if (this.openness !== null) {
        const halfGap = axisLength * (0.06 + this.openness * 0.38);
        for (const sign of [-1, 1]) {
          const root = position.map((v, i) => v + y[i] * sign * halfGap - x[i] * axisLength * 0.15);
          const tip = root.map((v, i) => v + x[i] * axisLength * 0.45);
          this.line(ctx, root, tip, width, height, '#f2f5fa', 3);
        }
      }
      const point = this.project(position, width, height);
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(...point, 5, 0, 2 * Math.PI); ctx.fill();
      this.arrow(ctx, position, x, axisLength, width, height, '#ff6767', 'X');
      this.arrow(ctx, position, y, axisLength, width, height, '#79db8b', 'Y');
      this.arrow(ctx, position, z, axisLength, width, height, '#77aaff', 'Z');
    }
    ctx.fillStyle = '#aabbd0'; ctx.font = '12px system-ui';
    ctx.fillText('固定视角 · 拖动旋转 · 滚轮缩放', 12, height - 14);
  }
}

/* Contract-only TCP pose display. Each arm is drawn in its own fixed reference frame. */
class Pose3D {
  constructor(canvas, stream, side) {
    this.canvas = canvas;
    this.stream = stream;
    this.side = side;
    this.yaw = -0.75;
    this.pitch = 0.42;
    this.zoom = 1;
    this.index = -2;
    this.openness = null;
    this.drag = null;
    this.bounds = this.getBounds();
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
    this.yaw = -0.75; this.pitch = 0.42; this.zoom = 1; this.draw();
  }

  getBounds() {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (const pose of this.stream.values) for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], pose[axis]);
      max[axis] = Math.max(max[axis], pose[axis]);
    }
    const center = min.map((v, i) => (v + max[i]) / 2);
    const span = Math.max(0.2, ...min.map((v, i) => max[i] - v));
    return {min, max, center, span};
  }

  setSample(index, openness) {
    if (this.index === index && this.openness === openness) return;
    this.index = index;
    this.openness = openness;
    this.draw();
  }

  project(point, width, height) {
    const p = point.map((v, i) => v - this.bounds.center[i]);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const scale = Math.min(width, height) * 0.62 / this.bounds.span * this.zoom;
    const horizontal = cy * p[0] - sy * p[1];
    const depth = sy * p[0] + cy * p[1];
    return [width / 2 + horizontal * scale, height / 2 + (sp * depth - cp * p[2]) * scale];
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

  drawPath(ctx, width, height, from, to, color, thickness) {
    if (to < from) return;
    const count = to - from + 1;
    const step = Math.max(1, Math.ceil(count / 1800));
    ctx.strokeStyle = color; ctx.lineWidth = thickness; ctx.beginPath();
    for (let i = from; i <= to; i += step) {
      const point = this.project(this.stream.values[i].slice(0, 3), width, height);
      if (i === from) ctx.moveTo(...point); else ctx.lineTo(...point);
    }
    if ((to - from) % step !== 0) ctx.lineTo(...this.project(
      this.stream.values[to].slice(0, 3), width, height));
    ctx.stroke();
  }

  draw() {
    const canvas = this.canvas, width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const dpi = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpi); canvas.height = Math.round(height * dpi);
    const ctx = canvas.getContext('2d'); ctx.scale(dpi, dpi);
    ctx.fillStyle = '#0c1520'; ctx.fillRect(0, 0, width, height);
    const {center, span} = this.bounds;
    const floor = center[2] - span * 0.45;
    for (let step = -2; step <= 2; step++) {
      const offset = step * span / 4;
      this.line(ctx, [center[0] + offset, center[1] - span / 2, floor],
        [center[0] + offset, center[1] + span / 2, floor], width, height, '#213346');
      this.line(ctx, [center[0] - span / 2, center[1] + offset, floor],
        [center[0] + span / 2, center[1] + offset, floor], width, height, '#213346');
    }
    this.drawPath(ctx, width, height, 0, this.stream.values.length - 1, '#40556b', 1.5);
    if (this.index >= 0) {
      this.drawPath(ctx, width, height, 0, this.index,
        this.side === 'left' ? '#74c7ff' : '#ffbd7f', 2.5);
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
    ctx.fillText('拖动旋转 · 滚轮缩放', 12, height - 14);
  }
}

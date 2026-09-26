/* Contract-only TCP pose display. Each arm keeps its own fixed reference frame. */
import * as THREE from './vendor/three.module.min.js';

export class Pose3D {
  constructor(canvas, stream, side) {
    this.canvas = canvas;
    this.stream = stream;
    this.side = side;
    // Look forward along the initial tool +X from behind and above it.
    // Opposite lateral offsets approximate looking from the head toward each hand.
    this.homeView = {yaw: Math.PI / 2 + (side === 'left' ? -0.28 : 0.28), pitch: 0.4, zoom: 1};
    Object.assign(this, this.homeView);
    this.index = -1;
    this.openness = null;
    this.drag = null;
    this.trailNs = 2e9;
    const initial = stream.values[0];
    const x = new THREE.Vector3().fromArray(initial, 3);
    const y = new THREE.Vector3().fromArray(initial, 6);
    const z = new THREE.Vector3().crossVectors(x, y);
    this.initialAxes = [x.toArray(), y.toArray(), z.toArray()];
    const frame = this.getEpisodeFrame();
    this.span = frame.span;
    this.axisLength = Math.max(0.055, Math.min(0.2, this.span * 0.18));
    this.radius = frame.radius + this.axisLength * 1.2;
    // A rigid change of basis: fixed reference -> initial tool axes, centered on the episode.
    this.viewFromReference = new THREE.Matrix4().makeBasis(x, y, z)
      .setPosition(...frame.center).invert();
    try {
      this.renderer = new THREE.WebGLRenderer({canvas, antialias: true});
    } catch (error) {
      throw new Error('Unable to display the 3D pose; check that the browser supports WebGL 2 and hardware acceleration is enabled.', {cause: error});
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor('#0c1520');
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, this.radius / 1000, this.radius * 100);
    this.camera.up.set(0, 0, 1);
    const fill = new THREE.HemisphereLight(0xe6f0ff, 0x394254, 1.8);
    fill.position.set(0, 0, 1);
    this.scene.add(fill);
    const key = new THREE.DirectionalLight(0xffffff, 3);
    key.position.set(-3, -4, 6);
    this.scene.add(key);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.createGeometry();
    this.message = document.createElement('span');
    this.message.className = 'pose-empty';
    this.message.textContent = 'Waiting for first TCP sample';
    canvas.parentElement.append(this.message);
    this.onDown = event => {
      if (event.button !== 0) return;
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
    canvas.addEventListener('lostpointercapture', this.onUp);
    canvas.addEventListener('wheel', this.onWheel, {passive: false});
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
    this.draw();
  }

  createGeometry() {
    const grid = new THREE.GridHelper(this.span * 1.3, 10, 0x35495e, 0x213346);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -this.span * 0.45;
    this.content.add(grid);
    this.tool = new THREE.Group();
    this.tool.matrixAutoUpdate = false;
    this.content.add(this.tool);
    const glyph = new THREE.Group();
    glyph.scale.setScalar(this.axisLength);
    this.tool.add(glyph);
    const shaftGeometry = new THREE.CylinderGeometry(0.025, 0.025, 0.76, 20);
    const headGeometry = new THREE.ConeGeometry(0.075, 0.24, 24);
    const axes = [
      {direction: new THREE.Vector3(1, 0, 0), color: '#ff6767', label: 'X'},
      {direction: new THREE.Vector3(0, 1, 0), color: '#79db8b', label: 'Y'},
      {direction: new THREE.Vector3(0, 0, 1), color: '#77aaff', label: 'Z'},
    ];
    for (const {direction, color, label} of axes) {
      const material = new THREE.MeshStandardMaterial({color, roughness: 0.4});
      const arrow = new THREE.Group();
      arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      const shaft = new THREE.Mesh(shaftGeometry, material);
      shaft.position.y = 0.38;
      const head = new THREE.Mesh(headGeometry, material);
      head.position.y = 0.88;
      arrow.add(shaft, head);
      glyph.add(arrow);
      const text = document.createElement('canvas');
      text.width = 128; text.height = 64;
      const ctx = text.getContext('2d');
      ctx.font = 'bold 44px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.strokeStyle = '#0c1520'; ctx.lineWidth = 5;
      ctx.strokeText(label, 64, 32);
      ctx.fillStyle = color; ctx.fillText(label, 64, 32);
      const map = new THREE.CanvasTexture(text);
      map.colorSpace = THREE.SRGBColorSpace;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map, sizeAttenuation: false,
        depthTest: true, depthWrite: false}));
      sprite.position.copy(direction).multiplyScalar(this.axisLength * 1.18);
      sprite.scale.set(0.085, 0.0425, 1);
      this.tool.add(sprite);
    }
    // The opaque sphere hides the roots of arrows pointing behind the TCP.
    const white = new THREE.MeshStandardMaterial({color: '#e9eef5', roughness: 0.5});
    const tcp = new THREE.Mesh(new THREE.SphereGeometry(0.12, 32, 20), white);
    glyph.add(tcp);
    this.fingers = [-1, 1].map(sign => {
      const finger = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.075, 0.1), white);
      finger.position.x = 0.075;
      finger.userData.sign = sign;
      glyph.add(finger);
      return finger;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(251 * 3), 3)
      .setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(251 * 3), 3)
      .setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.trail = new THREE.Line(geometry, new THREE.LineBasicMaterial({vertexColors: true}));
    this.trail.frustumCulled = false;
    this.content.add(this.trail);
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener('pointerdown', this.onDown);
    this.canvas.removeEventListener('pointermove', this.onMove);
    this.canvas.removeEventListener('pointerup', this.onUp);
    this.canvas.removeEventListener('pointercancel', this.onUp);
    this.canvas.removeEventListener('lostpointercapture', this.onUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.message.remove();
    const resources = new Set();
    this.scene.traverse(object => {
      if (object.geometry) resources.add(object.geometry);
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (!material) continue;
        if (material.map) resources.add(material.map);
        resources.add(material);
      }
    });
    resources.forEach(resource => resource.dispose());
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }

  reset() {
    Object.assign(this, this.homeView);
    this.draw();
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
    const ranges = min.map((value, i) => max[i] - value);
    return {
      center: origin.map((value, i) => value + this.initialAxes.reduce(
        (sum, axis, j) => sum + axis[i] * middle[j], 0)),
      span: Math.max(0.25, ...ranges) * 1.8,
      radius: Math.max(0.125, Math.hypot(...ranges) / 2),
    };
  }

  setSample(index, openness) {
    if (this.index === index && this.openness === openness) return;
    this.index = index;
    this.openness = openness;
    if (index >= 0) {
      const pose = this.stream.values[index];
      const x = new THREE.Vector3().fromArray(pose, 3);
      const y = new THREE.Vector3().fromArray(pose, 6);
      const z = new THREE.Vector3().crossVectors(x, y);
      const referenceFromTool = new THREE.Matrix4().makeBasis(x, y, z).setPosition(...pose.slice(0, 3));
      this.tool.matrix.multiplyMatrices(this.viewFromReference, referenceFromTool);
      for (const finger of this.fingers) {
        finger.visible = openness !== null;
        finger.position.y = finger.userData.sign * (0.06 + (openness ?? 0) * 0.38);
      }
      this.updateTrail();
    }
    this.draw();
  }

  updateTrail() {
    const from = this.trailStart(this.index), to = this.index;
    const step = Math.max(1, Math.ceil((to - from) / 250));
    const times = this.stream.times, values = this.stream.values;
    const duration = Math.max(1, times[to] - times[from]);
    const positions = this.trail.geometry.getAttribute('position');
    const colors = this.trail.geometry.getAttribute('color');
    const bright = new THREE.Color(this.side === 'left' ? '#74c7ff' : '#ffbd7f');
    const dark = new THREE.Color('#0c1520'), color = new THREE.Color();
    const point = new THREE.Vector3();
    let count = 0;
    for (let i = from;; i = Math.min(to, i + step)) {
      point.fromArray(values[i]).applyMatrix4(this.viewFromReference);
      positions.setXYZ(count, point.x, point.y, point.z);
      const age = (times[i] - times[from]) / duration;
      color.copy(dark).lerp(bright, 0.14 + 0.82 * age);
      colors.setXYZ(count, color.r, color.g, color.b);
      count++;
      if (i === to) break;
    }
    this.trail.geometry.setDrawRange(0, count);
    positions.needsUpdate = colors.needsUpdate = true;
  }

  draw() {
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    if (!width || !height) return;
    if (width !== this.width || height !== this.height) {
      this.width = width; this.height = height;
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const fitAngle = Math.min(halfFov, Math.atan(Math.tan(halfFov) * this.camera.aspect));
    const distance = this.radius * 1.1 / Math.sin(fitAngle) / this.zoom;
    // Explicit right-handed orbit camera with +Z up. Positive pitch is above the XY plane.
    this.camera.position.set(-Math.cos(this.pitch) * Math.sin(this.yaw),
      -Math.cos(this.pitch) * Math.cos(this.yaw), Math.sin(this.pitch)).multiplyScalar(distance);
    this.camera.lookAt(0, 0, 0);
    this.content.visible = this.index >= 0;
    this.message.hidden = this.index >= 0;
    this.renderer.render(this.scene, this.camera);
  }
}

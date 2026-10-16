// 3D Titanic viewer, used as a Streamlit component (see ship.py).
//
// Modes
//   berth: highlights where a passenger of a given class slept vs where the lifeboats were
//   sink:  pitches and sinks the ship to a given point in the timeline, with launched boats on the water
import * as THREE from 'three';
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { OrbitControls } from './vendor/OrbitControls.js';

// ── Ship anchors, in model units ───────────────────────────────────────────────
// The model is ~409 units long (≈1.52 units per metre), bow towards +x, keel at y = 0.
// Calibrated against the model: hull top (C deck) ≈ 35, boat deck ≈ 47, waterline at the
// 10.5 m draft ≈ 16. Deck positions are approximate, from the ship's deck plans.
const SHIP = { length: 409, waterline: 16 };
const ANCHORS = {
  boats_fwd:   [105, 50, 21],    // boat deck, around the first funnel
  boats_aft:   [-70, 50, 21],    // boat deck, around the third and fourth funnels
  first:       [40, 37, 0],      // B/C decks, amidships
  second:      [-100, 28, 0],    // D/E decks, aft
  third_bow:   [150, 21, 0],     // F/G decks, bow (mostly single men)
  third_stern: [-150, 23, 0],    // F/G decks, stern (mostly families and women)
};

// ── Streamlit component protocol (no build step needed) ──
const standalone = window.parent === window;
function post(type, data = {}) {
  if (!standalone) window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*');
}
const query = new URLSearchParams(location.search);
let args = {
  mode: query.get('mode') || 'berth',
  height: 420,
  markers: [
    { at: 'boats_fwd', title: 'Lifeboats', sub: 'boat deck', color: '#4ade80' },
    { at: 'third_bow', title: 'Your cabin', sub: '3rd class · lower decks, bow', color: '#f87171', pulse: true },
  ],
  pitch: +(query.get('pitch') || 0), sink: +(query.get('sink') || 0), boats: +(query.get('boats') || 0),
  autorotate: true,
};

// ── Scene ──
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, 1, 1, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 220;
controls.maxDistance = 1400;
controls.maxPolarAngle = Math.PI * 0.49;
controls.autoRotateSpeed = 0.6;

scene.add(new THREE.HemisphereLight(0xffffff, 0x1b2430, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(-300, 500, 400);
scene.add(sun);

// Sea: a soft, fading disc at the waterline
function radialTexture(inner, outer) {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, inner);
  grad.addColorStop(0.3, inner);
  grad.addColorStop(1, outer);
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const sea = new THREE.Mesh(
  new THREE.CircleGeometry(700, 64),
  new THREE.MeshBasicMaterial({ map: radialTexture('rgba(30,42,56,0.9)', 'rgba(30,42,56,0)'),
                                transparent: true, depthWrite: false }),
);
sea.rotation.x = -Math.PI / 2;
sea.position.y = SHIP.waterline;
sea.renderOrder = 2;
scene.add(sea);

// Ship pivots around its midpoint at the waterline so pitching looks natural
const pivot = new THREE.Group();
pivot.position.y = SHIP.waterline;
scene.add(pivot);
const ship = new THREE.Group();
ship.position.y = -SHIP.waterline;
pivot.add(ship);

// Lifeboats on the water (sink mode)
const boatGeo = new THREE.BoxGeometry(9, 2, 3);
const boatMat = new THREE.MeshStandardMaterial({ color: 0xe8e2d0, roughness: 0.9 });
const boats = [];
for (let i = 0; i < 20; i++) {
  const b = new THREE.Mesh(boatGeo, boatMat);
  const side = i % 2 ? 1 : -1;
  const row = Math.floor(i / 2);
  b.position.set(-120 + row * 34 + (i % 3) * 6, SHIP.waterline + 0.6, side * (60 + (i * 37) % 90));
  b.rotation.y = (i * 1.7) % 1 - 0.5;
  b.visible = false;
  b.scale.setScalar(0.001);
  b.userData.target = 0;
  scene.add(b);
  boats.push(b);
}

// ── Markers ──
const glow = radialTexture('rgba(255,255,255,1)', 'rgba(255,255,255,0)');
let markers = [];
function setMarkers(list) {
  for (const m of markers) { ship.remove(m.group); m.tag.remove(); }
  markers = (list || []).map((spec) => {
    const group = new THREE.Group();
    group.position.fromArray(ANCHORS[spec.at] || [0, 40, 0]);
    const color = new THREE.Color(spec.color || '#ffffff');
    const dot = new THREE.Mesh(new THREE.SphereGeometry(3.2, 20, 12),
                               new THREE.MeshBasicMaterial({ color, depthTest: false }));
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color, transparent: true,
                                                             depthTest: false, opacity: 0.6 }));
    halo.scale.setScalar(26);
    dot.renderOrder = halo.renderOrder = 10;
    group.add(halo, dot);
    ship.add(group);
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.innerHTML = `<span style="color:${spec.color}">●</span> ${spec.title || ''}<small>${spec.sub || ''}</small>`;
    if (spec.title) stage.appendChild(tag);
    return { group, halo, tag, pulse: !!spec.pulse };
  });
}

// ── Args → state ──
const target = { pitch: 0, sink: 0 };
function apply(a) {
  args = { ...args, ...a };
  post('streamlit:setFrameHeight', { height: args.height });
  setMarkers(args.mode === 'berth' ? args.markers : []);
  target.pitch = THREE.MathUtils.degToRad(args.mode === 'sink' ? args.pitch : 0);
  target.sink = args.mode === 'sink' ? args.sink : 0;
  boats.forEach((b, i) => { b.userData.target = args.mode === 'sink' && i < args.boats ? 1 : 0; });
  controls.autoRotate = !!args.autorotate;
  resize();
}

window.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'streamlit:render') apply(e.data.args || {});
});

// ── Load model ──
new GLTFLoader().load('./titanic.glb', (gltf) => {
  gltf.scene.traverse((o) => {
    if (o.isMesh) {
      o.material.color.convertSRGBToLinear();
      o.material.side = THREE.DoubleSide;
    }
  });
  ship.add(gltf.scene);
  document.getElementById('loading').remove();
}, undefined, (err) => {
  document.getElementById('loading').textContent = 'could not load the 3D model';
  console.error(err);
});

// ── Camera ──
camera.position.set(340, 140, 360);
controls.target.set(0, 30, 0);
if (query.get('cam')) {   // standalone debugging: ?cam=x,y,z
  camera.position.fromArray(query.get('cam').split(',').map(Number));
  if (query.get('target')) controls.target.fromArray(query.get('target').split(',').map(Number));
  if (query.get('fov')) camera.userData.fov = +query.get('fov');
  args.autorotate = false;
}
if (query.has('debug')) {  // ?debug: hide the sea, draw a 10-unit grid on the ship's centre plane
  document.body.style.background = '#161616';
  sea.visible = false;
  const pts = [];
  for (let y = 0; y <= 110; y += 10) pts.push(-220, y, 0, 220, y, 0);
  for (let x = -200; x <= 200; x += 50) pts.push(x, 0, 0, x, 110, 0);
  const g = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xff00ff, depthTest: false })));
}

function resize() {
  const w = stage.clientWidth || window.innerWidth;
  const h = args.height || window.innerHeight;
  stage.style.height = `${h}px`;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  // keep the whole ship in frame on narrow screens
  camera.fov = camera.userData.fov || (w / h < 1.4 ? 40 : 28);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ── Loop (paused while off screen) ──
let visible = true;
new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(stage);
const tmp = new THREE.Vector3();
const clock = new THREE.Clock();

function tick() {
  requestAnimationFrame(tick);
  if (!visible) return;
  const t = clock.getElapsedTime();
  const k = 0.06;
  pivot.rotation.z += (-target.pitch - pivot.rotation.z) * k;   // bow is +x, so -z rotation dips it
  pivot.position.y += (SHIP.waterline - target.sink - pivot.position.y) * k;
  for (const b of boats) {
    b.visible = b.userData.target > 0 || b.scale.x > 0.02;
    const s = b.scale.x + (b.userData.target - b.scale.x) * 0.08;
    b.scale.setScalar(Math.max(s, 0.001));
    b.position.y = SHIP.waterline + 0.6 + Math.sin(t * 1.3 + b.position.x) * 0.4;
  }
  const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  for (const m of markers) {
    if (m.pulse) m.halo.scale.setScalar(22 + Math.sin(t * 3) * 7);
    m.group.getWorldPosition(tmp).project(camera);
    m.tag.style.left = `${(tmp.x * 0.5 + 0.5) * w}px`;
    m.tag.style.top = `${(-tmp.y * 0.5 + 0.5) * h}px`;
    m.tag.style.opacity = tmp.z < 1 ? 1 : 0;
  }
  controls.update();
  renderer.render(scene, camera);
}

apply({});
post('streamlit:componentReady', { apiVersion: 1 });
tick();

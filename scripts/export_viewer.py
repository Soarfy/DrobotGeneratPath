"""Export a planned trajectory into the standalone browser viewer.

Writes out/viewer_data.json and copies the viewer template next to it, so the
result is a self-contained folder you can open in any browser (no server, no
Isaac Sim, no ROS).

    python scripts/export_viewer.py --trajectory out/arch_lower_jaw.json
    python scripts/export_viewer.py --trajectory out/trajectory.json --out out/viewer_raster

Then open out/viewer.html, or serve the folder:

    cd out && python -m http.server 8765
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen.config import Config  # noqa: E402
from dobot_pathgen.mesh_io import read_ply, read_stl  # noqa: E402
from dobot_pathgen.planning import Trajectory  # noqa: E402
from dobot_pathgen.urdf_chain import build_chain  # noqa: E402


# The URDF's fixed world->dummy->base chain means fk_all() returns one frame
# per joint entry, in this order.
LINK_ORDER = ["base_link", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6"]
MESH_STEMS = ["base_link", "J1", "J2", "J3", "J4", "J5", "J6"]

VIEWER_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CR5A 轨迹回放</title>
<style>
  html,body{margin:0;height:100%;background:#14181c;color:#e6e6e6;
            font-family:system-ui,"Microsoft YaHei",sans-serif;overflow:hidden}
  #ui{position:fixed;left:14px;top:14px;background:rgba(10,14,18,.82);
      padding:12px 16px;border-radius:10px;font-size:13px;line-height:1.7;
      backdrop-filter:blur(6px);border:1px solid #2a323b;min-width:250px}
  #ui b{color:#7bd389} #ui .dim{color:#8b95a0}
  input[type=range]{width:100%;margin:6px 0}
  button{background:#243140;color:#dfe7ee;border:1px solid #35465a;
         border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}
  button:hover{background:#2e3f52}
  #legend{position:fixed;right:14px;top:14px;background:rgba(10,14,18,.82);
          padding:10px 14px;border-radius:10px;font-size:12px;line-height:1.9;
          border:1px solid #2a323b}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
</style>
</head>
<body>
<div id="ui">
  <div>CR5A 轨迹回放 <span class="dim">__TITLE__</span></div>
  <div>状态：<b id="state">就绪</b></div>
  <div class="dim">时间 <span id="time">0.0</span> / <span id="total">0</span> s</div>
  <div class="dim">路点 <span id="wp">0</span> / <span id="wpmax">0</span></div>
  <input id="scrub" type="range" min="0" max="100" value="0">
  <div><button id="play">播放</button>
       <button id="reset">重播</button>
       <button id="loop">循环: 开</button></div>
  <div class="dim" style="margin-top:8px">鼠标拖动旋转 / 滚轮缩放</div>
</div>
<div id="legend">
  <div><span class="sw" style="background:#6b9bd4"></span>机械臂</div>
  <div><span class="sw" style="background:#f2f4f7"></span>刷棒</div>
  <div><span class="sw" style="background:#e0d0ba"></span>牙模</div>
  <div><span class="sw" style="background:#4caf50"></span>轨迹</div>
</div>
<script type="importmap">
{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const data = await (await fetch('viewer_data.json')).json();
const frames = data.frames;
const n = frames.length;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14181c);
const camera = new THREE.PerspectiveCamera(32, innerWidth/innerHeight, 0.01, 30);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
{
  // Frame the whole motion envelope: brush tips, flanges and the tooth model.
  // The arm base is deliberately excluded -- it is far away and would shrink
  // the interesting part to a few pixels.
  const lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
  const inc = p => { for(let k=0;k<3;k++){ if(p[k]<lo[k])lo[k]=p[k]; if(p[k]>hi[k])hi[k]=p[k]; } };
  // Frame the brush and the teeth only. Including every flange position pulls
  // in the whole arm and shrinks the brushing area to a few pixels.
  for (const f of frames) { inc(f.tip); }
  if (data.toolMesh) for (const p of data.toolMesh.v) inc(p);
  for (const p of data.targetMesh.v) inc(p);
  const c = lo.map((v,k)=>(v+hi[k])*0.5);
  const span = Math.max(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]);
  controls.target.set(c[0], c[1], c[2]);
  // span already covers tip->flange, so 1.35x puts the camera just outside the
  // envelope; 2.0x leaves room to read the brush contact and the wrist together.
  const dist = span * 1.05;
  // Look along -Y so the brush body does not hide the tooth contact.
  const dir = new THREE.Vector3(-0.62, -0.52, 0.59).normalize();
  camera.position.copy(controls.target.clone().add(dir.multiplyScalar(dist)));
  camera.lookAt(controls.target);
  controls.update();
}

scene.add(new THREE.AmbientLight(0xffffff, 1.4));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(1.5,-1.8,2.2); scene.add(key);
const fill = new THREE.DirectionalLight(0x9fc4e8, 0.9);
fill.position.set(-1.2,1.4,0.8); scene.add(fill);

// target
const TGT = data.targetMesh;
const tg = new THREE.BufferGeometry();
tg.setAttribute('position', new THREE.Float32BufferAttribute(TGT.v.flat(), 3));
tg.setIndex(TGT.f);
tg.computeVertexNormals();
scene.add(new THREE.Mesh(tg, new THREE.MeshStandardMaterial({
  color:0xe0d0ba, metalness:0.05, roughness:0.75, side:THREE.DoubleSide
})));

// arm
const linkMeshes = {};
for (const name of data.linkOrder) {
  const L = data.links[name];
  if (!L) continue;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(L.v.flat(), 3));
  g.setIndex(L.f);
  g.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: name === 'base_link' ? 0x39434e : 0x6b9bd4,
    metalness:0.15, roughness:0.62, side:THREE.DoubleSide
  });
  const m = new THREE.Mesh(g, mat);
  m.matrixAutoUpdate = false;   // pose comes from an explicit matrix each frame
  scene.add(m);
  linkMeshes[name] = m;
}

// brush: real mesh in the Link6 frame when exported, else a plain cylinder
let brush;
if (data.toolMesh) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(data.toolMesh.v.flat(), 3));
  g.setIndex(data.toolMesh.f);
  g.computeVertexNormals();
  brush = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
    color:0xf4f6f8, side:THREE.DoubleSide
  }));
  brush.matrixAutoUpdate = false;
} else {
  brush = new THREE.Mesh(
    new THREE.CylinderGeometry(data.tool.radius, data.tool.radius, 1, 20),
    new THREE.MeshStandardMaterial({color:0xf2f4f7, metalness:0.1, roughness:0.5})
  );
  brush.geometry.translate(0, 0.5, 0);
}
scene.add(brush);

const tipMark = new THREE.Mesh(
  new THREE.SphereGeometry(0.008, 16, 16),
  new THREE.MeshStandardMaterial({color:0xffb703, emissive:0x8a5f00})
);
scene.add(tipMark);

// trajectory
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.Float32BufferAttribute(frames.map(f=>f.tip).flat(), 3));
scene.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({color:0x4caf50})));

const total = frames[n-1].t;
const ui = {
  state: document.getElementById('state'),
  time: document.getElementById('time'),
  total: document.getElementById('total'),
  wp: document.getElementById('wp'),
  wpmax: document.getElementById('wpmax'),
  scrub: document.getElementById('scrub'),
};
ui.total.textContent = total.toFixed(1);
ui.wpmax.textContent = n;
ui.scrub.max = n - 1;

function applyFrame(i) {
  i = Math.max(0, Math.min(n-1, Math.round(i)));
  const f = frames[i];
  for (let k = 0; k < data.linkOrder.length; k++) {
    const mesh = linkMeshes[data.linkOrder[k]];
    const T = f.frames[k];
    if (!mesh || !T) continue;
    mesh.matrix.set(
      T[0][0],T[0][1],T[0][2],T[0][3],
      T[1][0],T[1][1],T[1][2],T[1][3],
      T[2][0],T[2][1],T[2][2],T[2][3],
      0,0,0,1
    );
    mesh.matrixWorldNeedsUpdate = true;
  }
  const flange = new THREE.Vector3(...f.flange);
  const tip = new THREE.Vector3(...f.tip);
  const T6 = f.frames[6];
  if (data.toolMesh) {
    brush.matrix.set(
      T6[0][0],T6[0][1],T6[0][2],T6[0][3],
      T6[1][0],T6[1][1],T6[1][2],T6[1][3],
      T6[2][0],T6[2][1],T6[2][2],T6[2][3],
      0,0,0,1
    );
    brush.matrixWorldNeedsUpdate = true;
  } else {
    const dir = tip.clone().sub(flange);
    brush.position.copy(flange);
    brush.scale.set(1, dir.length(), 1);
    brush.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), dir.clone().normalize());
  }
  tipMark.position.copy(tip);
  ui.time.textContent = f.t.toFixed(1);
  ui.wp.textContent = i + 1;
  ui.scrub.value = i;
}
applyFrame(0);

let idx = 0, playing = false, loop = true, acc = 0, last = performance.now();
document.getElementById('play').onclick = () => {
  playing = !playing;
  document.getElementById('play').textContent = playing ? '暂停' : '播放';
  ui.state.textContent = playing ? '运行中' : '暂停';
  last = performance.now();
};
document.getElementById('reset').onclick = () => { acc = 0; idx = 0; applyFrame(0); };
document.getElementById('loop').onclick = e => {
  loop = !loop; e.target.textContent = '循环: ' + (loop ? '开' : '关');
};
ui.scrub.oninput = e => { idx = +e.target.value; applyFrame(idx); };

function animate(now) {
  requestAnimationFrame(animate);
  if (playing) {
    acc += (now - last) / 1000 * 3.0;   // playback speed multiplier
    last = now;
    if (acc >= total) {
      if (loop) { acc = 0; }
      else { acc = total; playing = false;
             document.getElementById('play').textContent = '播放';
             ui.state.textContent = '完成'; }
    }
    let lo = 0, hi = n - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; (frames[mid].t < acc) ? lo = mid + 1 : hi = mid; }
    if (lo !== idx) { idx = lo; applyFrame(idx); }
  }
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>
"""


def export(cfg, trajectory_path, out_dir: Path, decimate: int = 1, title: str = ""):
    trajectory = Trajectory.load(trajectory_path)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)

    tcp = np.eye(4)
    tcp[:3, 3] = cfg.tool.tcp_offset_m

    frames = []
    for w in trajectory.waypoints:
        fk = chain.fk_all(w.positions)
        T6 = fk[6]
        frames.append(
            {
                "t": round(w.time_s, 3),
                "flange": [round(float(v), 5) for v in T6[:3, 3]],
                "tip": [round(float(v), 5) for v in (T6 @ tcp)[:3, 3]],
                "q": [round(float(np.degrees(v)), 2) for v in w.positions],
                "frames": [
                    [[round(float(x), 5) for x in row] for row in fk[k].tolist()]
                    for k in range(7)
                ],
            }
        )

    payload = {
        "frames": frames,
        "linkOrder": LINK_ORDER,
        "tool": {
            "radius": cfg.tool.radius_m,
            "offset": [float(v) for v in cfg.tool.tcp_offset_m],
        },
        "links": {},
    }

    mesh_dir = cfg.robot.urdf.parent.parent / "meshes" / "cr5"
    for link, stem in zip(LINK_ORDER, MESH_STEMS):
        path = mesh_dir / f"{stem}.STL"
        if not path.exists():
            path = mesh_dir / f"{stem}.stl"
        verts, faces, _ = read_stl(path)
        if decimate > 1:
            faces = faces[::decimate]
            used = np.unique(faces)
            remap = {int(o): i for i, o in enumerate(used)}
            verts = verts[used]
            faces = np.array([[remap[int(i)] for i in tri] for tri in faces])
        payload["links"][link] = {
            "v": [[round(float(c), 5) for c in p] for p in verts],
            "f": [int(i) for tri in faces for i in tri],
        }

    vertices, faces, _, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    vertices = vertices @ cfg.scene.model_to_base[:3, :3].T + cfg.scene.model_to_base[:3, 3]
    payload["targetMesh"] = {
        "v": [[round(float(c), 5) for c in p] for p in vertices],
        "f": [int(i) for tri in faces for i in tri],
    }

    # Real brush geometry, already expressed in the Link6 frame, so the viewer
    # can pose it with the same matrix it uses for Link6.
    brush_stl = ROOT / "assets/robot/cr5a_with_tool/meshes/brushleap.stl"
    if brush_stl.exists():
        bverts, bfaces, _ = read_stl(brush_stl)
        payload["toolMesh"] = {
            "v": [[round(float(c), 5) for c in p] for p in bverts],
            "f": [int(i) for tri in bfaces for i in tri],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "viewer_data.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")

    label = title or Path(trajectory_path).stem
    html = VIEWER_TEMPLATE.replace("__TITLE__", label)
    (out_dir / "viewer.html").write_text(html, encoding="utf-8")

    return data_path, out_dir / "viewer.html", len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--trajectory", default=str(ROOT / "out/arch_lower_jaw.json"))
    parser.add_argument("--out", default=str(ROOT / "out/viewer"))
    # Dropping triangles by index leaves holes in the surface, so the default
    # keeps the full mesh. Raise this only if the JSON size becomes a problem.
    parser.add_argument("--decimate", type=int, default=1,
                        help="keep every Nth triangle of each arm link (1 = all)")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    data_path, html_path, n = export(
        cfg, args.trajectory, Path(args.out), decimate=args.decimate, title=args.title
    )
    size_mb = data_path.stat().st_size / 1e6
    print(f"frames   : {n}")
    print(f"data     : {data_path} ({size_mb:.1f} MB)")
    print(f"viewer   : {html_path}")
    print("\nOpen the viewer:")
    print(f'  cd {Path(args.out).resolve()} && python -m http.server 8765')
    print("  http://127.0.0.1:8765/viewer.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

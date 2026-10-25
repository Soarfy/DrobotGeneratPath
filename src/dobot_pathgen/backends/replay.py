"""Offline preview: write a PLY of the tool path and a self-contained HTML viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..mesh_io import read_ply, write_ply_points
from ..planning import Trajectory


def write_path_preview(cfg, trajectory: Trajectory, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces, _, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale

    tips = np.array([np.asarray(w.tool_pose)[:3, 3] for w in trajectory.waypoints])
    # Tool poses are stored in the base frame already.
    tip_local = (tips - cfg.scene.model_to_base[:3, 3]) @ cfg.scene.model_to_base[:3, :3]

    cloud = np.vstack([vertices, tips])
    colors = np.vstack(
        [
            np.tile(np.array([[170, 170, 170]], dtype=np.uint8), (len(vertices), 1)),
            np.tile(np.array([[230, 60, 40]], dtype=np.uint8), (len(tips), 1)),
        ]
    )
    ply_path = out_dir / "path_preview.ply"
    write_ply_points(ply_path, cloud, colors=colors)

    html = _html(tip_local, vertices, trajectory)
    html_path = out_dir / "path_preview.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _html(tips: np.ndarray, mesh_vertices: np.ndarray, trajectory: Trajectory) -> str:
    """Minimal Three.js scene: target mesh as a point cloud plus the tool path."""
    step = max(1, len(mesh_vertices) // 40000)
    pts = mesh_vertices[::step]
    payload = {
        "path": tips.round(5).tolist(),
        "target": pts.round(5).tolist(),
        "duration": trajectory.duration,
        "points": len(tips),
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CR5A path preview</title>
<style>
  html,body {{ margin:0; height:100%; background:#111418; color:#e6e6e6;
              font-family: system-ui, sans-serif; }}
  #hud {{ position:fixed; top:12px; left:12px; font-size:13px; line-height:1.5;
          background:rgba(0,0,0,.45); padding:10px 14px; border-radius:8px; }}
  #hud b {{ color:#ff6b4a; }}
</style>
</head>
<body>
<div id="hud">target: <b>{len(pts)}</b> pts &nbsp;|&nbsp; path: <b>{len(tips)}</b> waypoints
  &nbsp;|&nbsp; {trajectory.duration:.1f} s</div>
<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                 "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const data = {json.dumps(payload)};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.001, 10);
camera.position.set(0.12, 0.12, 0.18);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.02, 0);
controls.update();
scene.add(new THREE.AmbientLight(0xffffff, 1.4));

const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.Float32BufferAttribute(data.target.flat(), 3));
const mat = new THREE.PointsMaterial({{color:0x9aa3ad, size:0.0006}});
scene.add(new THREE.Points(geom, mat));

const lineGeom = new THREE.BufferGeometry();
lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(data.path.flat(), 3));
scene.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({{color:0xff4d3d}})));

const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.0022, 16, 16),
                              new THREE.MeshBasicMaterial({{color:0xffd166}}));
scene.add(sphere);
let i = 0;
function animate() {{
  requestAnimationFrame(animate);
  const p = data.path[i % data.path.length];
  sphere.position.set(p[0], p[1], p[2]);
  if (i % 2 === 0) i++;
  renderer.render(scene, camera);
}}
animate();
addEventListener('resize', () => {{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
</script>
</body>
</html>"""

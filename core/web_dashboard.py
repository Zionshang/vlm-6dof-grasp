"""Local read-only dashboard for robot-test logs, images and 3D results."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import re
import sys
import threading
import time
import webbrowser

import numpy as np

from visualization_data import grasp_geometries, point_cloud_arrays


_FLOW_PREFIXES = (
    "[任务]", "[流程]", "[发送]", "[到达]", "[就绪]", "[失败]", "[成功]", "[测试]",
    "[Safety]", "[Config]",
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _LogWriter:
    def __init__(self, dashboard, original, channel=None, echo=True):
        self.dashboard, self.original = dashboard, original
        self.channel, self.echo, self.buffer = channel, echo, ""

    def write(self, text):
        if self.echo:
            self.original.write(text)
            self.original.flush()
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            try:
                self.dashboard.log(line, self.channel)
            except Exception:
                pass
        return len(text)

    def flush(self):
        if self.echo:
            self.original.flush()


class WebDashboard:
    """Threaded localhost dashboard; it never issues robot commands."""

    def __init__(self, output_dir, host="127.0.0.1", port=8765,
                 open_browser=True, max_points=12000):
        self.output_dir = Path(output_dir).resolve()
        self.host, self.requested_port = host, int(port)
        self.open_browser, self.max_points = bool(open_browser), int(max_points)
        self.started_at = time.time() - 1.0
        self._lock = threading.Lock()
        self._logs = {"flow": [], "detail": []}
        self._task = {}
        self._scene, self._scene_version = {}, 0
        self._server = None
        self.start()

    def set_output_dir(self, output_dir):
        self.output_dir = Path(output_dir).resolve()

    def set_task(self, name, target):
        with self._lock:
            self._task = {"name": name, "target": target}

    def start(self):
        if self._server is not None:
            return self
        from flask import Flask, Response, jsonify, send_from_directory
        from plotly.offline import get_plotlyjs
        from werkzeug.serving import WSGIRequestHandler, make_server

        app = Flask(__name__)

        @app.get("/")
        def index():
            return Response(_HTML, mimetype="text/html")

        @app.get("/plotly.js")
        def plotly_js():
            return Response(get_plotlyjs(), mimetype="text/javascript")

        @app.get("/api/status")
        def status():
            with self._lock:
                payload = {**self._logs, "task": self._task,
                           "scene_version": self._scene_version}
            payload["images"] = self._images()
            return jsonify(payload)

        @app.get("/api/scene")
        def scene():
            with self._lock:
                return jsonify(self._scene)

        @app.get("/files/<path:name>")
        def output_file(name):
            return send_from_directory(self.output_dir, name)

        class QuietHandler(WSGIRequestHandler):
            def log_request(self, *args, **kwargs):
                pass

        try:
            self._server = make_server(self.host, self.requested_port, app,
                                       threaded=True, request_handler=QuietHandler)
        except OSError:
            self._server = make_server(self.host, 0, app, threaded=True,
                                       request_handler=QuietHandler)
        self.url = f"http://{self.host}:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        message = f"[流程] Web dashboard: {self.url}"
        print(message)
        self.log(message, "flow")
        if self.open_browser:
            threading.Thread(target=webbrowser.open, args=(self.url,), daemon=True).start()
        return self

    def log(self, line, channel=None):
        line = _ANSI.sub("", line).strip()
        if not line:
            return
        channel = channel or ("flow" if line.startswith(_FLOW_PREFIXES) else "detail")
        with self._lock:
            self._logs["flow"] = (self._logs["flow"] + [line])[-2000:]
            if channel == "detail":
                self._logs["detail"] = (self._logs["detail"] + [line])[-2000:]

    @contextmanager
    def _capture(self, channel, echo):
        stdout, stderr = sys.stdout, sys.stderr
        while isinstance(stdout, _LogWriter):
            stdout = stdout.original
        while isinstance(stderr, _LogWriter):
            stderr = stderr.original
        with redirect_stdout(_LogWriter(self, stdout, channel, echo)), \
                redirect_stderr(_LogWriter(self, stderr, "detail", echo)):
            yield

    def capture_output(self):
        return self._capture(None, True)

    def details(self):
        return self._capture("detail", True)

    def update_scene(self, color, depth, grasps, intrinsic):
        points, colors = point_cloud_arrays(
            color, depth, np.asarray(intrinsic), self.max_points,
        )
        meshes = []
        for geometry in grasp_geometries(grasps):
            vertices = np.asarray(geometry.vertices)
            triangles = np.asarray(geometry.triangles)
            meshes.append({"vertices": vertices.round(6).tolist(),
                           "triangles": triangles.tolist()})
        rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
        scene = {
            "points": points.round(6).tolist(),
            "colors": [f"rgb({r},{g},{b})" for r, g, b in rgb],
            "meshes": meshes,
        }
        with self._lock:
            self._scene = scene
            self._scene_version += 1

    def _images(self):
        if not self.output_dir.exists():
            return []
        images = []
        for path in self.output_dir.rglob("*"):
            if (not path.is_file() or path.suffix.lower() not in
                    {".jpg", ".jpeg", ".png"} or
                    path.stat().st_mtime < self.started_at):
                continue
            relative = path.relative_to(self.output_dir).as_posix()
            order, group = next(((order, label) for order, prefix, label in (
                (1, "2D_grasp/first_select/", "First 最终结果"),
                (0, "2D_grasp/", "筛选通过的 2D Grasps"),
                (2, "vlm/origin_", "二维几何筛除"),
                (3, "captures/", "RGB-D 拍摄"),
                (4, "sam/", "SAM Mask"),
            ) if relative.startswith(prefix)), (5, "VLM 检测"))
            images.append({"path": relative, "group": group, "order": order,
                           "mtime": path.stat().st_mtime})
        return sorted(images, key=lambda item: (item["order"], item["mtime"]))

    def close(self):
        if self._server is not None:
            time.sleep(1.0)  # allow the browser's final poll/file requests
            self._server.shutdown()
            self._server = None


_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Piper Grasp Monitor</title><script src="/plotly.js"></script>
<style>
:root{color-scheme:dark;--bg:#0b1017;--panel:#131b25;--line:#263443;--text:#dce7f2;--accent:#38bdf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
header{height:58px;padding:10px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;font-weight:700;color:var(--accent)}
.task{display:flex;gap:8px;margin-left:auto}.badge{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text)}
main{display:grid;grid-template-rows:32vh 1fr;gap:10px;padding:10px;height:calc(100vh - 58px)}
.logs,.lower{display:grid;grid-template-columns:1fr 1fr;gap:10px;min-height:0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-height:0;overflow:hidden}
.title{padding:8px 11px;border-bottom:1px solid var(--line);font-weight:650}.terminal{height:calc(100% - 36px);padding:10px;overflow:auto;white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;color:#b9f6ca}
#detail{color:#c8d5e3}.images{height:calc(100% - 36px);overflow:auto;padding:8px}.group{margin:4px 0 12px}.group h3{font-size:12px;color:#8fcde8;margin:5px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:8px}.card{background:#0b1119;border:1px solid var(--line);padding:5px;border-radius:6px}.card img{width:100%;height:360px;object-fit:contain}.card div{font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#scene{width:100%;height:calc(100% - 36px)}@media(max-width:900px){main{height:auto;grid-template-rows:auto auto}.logs,.lower{grid-template-columns:1fr}.panel{height:42vh}}
</style></head><body><header>Piper D405 · FFS · EconomicGrasp Monitor<div id="task" class="task"></div></header><main>
<section class="logs"><div class="panel"><div class="title">总流程</div><div id="flow" class="terminal"></div></div><div class="panel"><div class="title">组件与筛选细节</div><div id="detail" class="terminal"></div></div></section>
<section class="lower"><div class="panel"><div class="title">O3D 同源点云与抓取姿态</div><div id="scene"></div></div><div class="panel"><div class="title">本次运行图像</div><div id="images" class="images"></div></div></section>
</main><script>
let sceneVersion=-1;
const esc=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function terminal(id,lines){const el=document.getElementById(id),bottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20;el.textContent=lines.join('\n');if(bottom)el.scrollTop=el.scrollHeight}
function renderTask(t){document.getElementById('task').innerHTML=t?.name?`<span class="badge">任务：${esc(t.name)}</span><span class="badge">目标：${esc(t.target)}</span>`:''}
function renderImages(items){const groups={};items.forEach(x=>(groups[x.group]??=[]).push(x));document.getElementById('images').innerHTML=Object.entries(groups).map(([g,xs])=>`<div class="group"><h3>${esc(g)}</h3><div class="grid">${xs.map(x=>`<div class="card"><img src="/files/${x.path.split('/').map(encodeURIComponent).join('/')}?v=${x.mtime}"><div title="${esc(x.path)}">${esc(x.path)}</div></div>`).join('')}</div></div>`).join('')}
async function renderScene(){const s=await (await fetch('/api/scene')).json(),traces=[];if(s.points?.length){traces.push({type:'scatter3d',mode:'markers',x:s.points.map(p=>p[0]),y:s.points.map(p=>p[1]),z:s.points.map(p=>p[2]),marker:{size:1.5,color:s.colors},name:'RGB-D'})}for(const m of s.meshes||[]){traces.push({type:'mesh3d',x:m.vertices.map(p=>p[0]),y:m.vertices.map(p=>p[1]),z:m.vertices.map(p=>p[2]),i:m.triangles.map(t=>t[0]),j:m.triangles.map(t=>t[1]),k:m.triangles.map(t=>t[2]),color:'#111827',flatshading:true,showscale:false,name:'grasp'})}Plotly.react('scene',traces,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#131b25',plot_bgcolor:'#131b25',font:{color:'#dce7f2'},scene:{aspectmode:'data',xaxis:{title:'X'},yaxis:{title:'Y'},zaxis:{title:'Z'},camera:{up:{x:0,y:-1,z:0}}}},{responsive:true,displaylogo:false})}
async function poll(){try{const s=await (await fetch('/api/status')).json();renderTask(s.task);terminal('flow',s.flow);terminal('detail',s.detail);renderImages(s.images);if(s.scene_version!==sceneVersion){sceneVersion=s.scene_version;await renderScene()}}catch(e){}setTimeout(poll,250)}poll();
</script></body></html>"""

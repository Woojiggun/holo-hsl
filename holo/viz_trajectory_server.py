"""Interactive: type text -> watch the model's hidden-state COMET fly through its byte universe.

The byte-stars are static (the map). When you type a sentence, the server runs the model, takes
the per-position hidden state at the universe's layer, projects it with the SAME PCA basis the
universe was built with, and the page animates a glowing comet tracing that path star-to-star.
That trajectory is the REAL motion (how the model's state moves as it reads), not decoration.

Reads universe_assets.json (stars + basis + which checkpoint/layer/arch). Loads that exact model.
Repoint to the new encoder by regenerating universe_assets.json from the AsymHSL checkpoint.

Run:  python viz_trajectory_server.py            then open http://127.0.0.1:8778
"""
from __future__ import annotations
import sys, json, pathlib, io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, torch
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_decoder import HSLDecoder
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASSETS = json.loads((HERE / "universe_assets.json").read_text(encoding="utf-8"))
B = ASSETS["basis"]
CENTER = np.array(B["center"], np.float32); COMPS = np.array(B["comps"], np.float32); SCALE = float(B["scale"])
LAYER = int(B["layer"]); ARCH = B.get("arch", "decoder")
print(f"basis: arch={ARCH} layer={LAYER} ckpt={pathlib.Path(B['ckpt']).name}")

ck = torch.load(B["ckpt"], map_location=DEV)
if ARCH == "asym":
    from hsl_asym import AsymHSL, pack_input, out_features
    cfg = ck.get("config", {})
    MODEL = AsymHSL(dim=B["dim"], enc_layers=cfg.get("enc_layers", 4), dec_layers=B["layers"], heads=B["heads"], K=cfg.get("K", 8)).to(DEV).eval()
    BLOCKS = MODEL.self_blocks; KK = cfg.get("K", 8)
else:
    MODEL = HSLDecoder(B["dim"], B["layers"], B["heads"]).to(DEV).eval()
    BLOCKS = MODEL.blocks; KK = None
MODEL.load_state_dict(ck["model"] if "model" in ck else ck)
_acts = {}
for i, blk in enumerate(BLOCKS):
    blk.register_forward_hook((lambda i: (lambda _m, _i, o: _acts.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())))(i))


@torch.no_grad()
def _read_points(data: bytes):
    """run a read pass over `data`, return per-byte projected points (same frame as stars)."""
    if ARCH == "asym":
        from hsl_asym import pack_input, out_features
        inpf = pack_input(data, KK).unsqueeze(0).to(DEV); of = out_features(data).unsqueeze(0).to(DEV)
        _ = MODEL(inpf, of); H = _acts[LAYER][0].float().cpu().numpy()
    else:
        f, _ = signal_features(data)
        feats = f.unsqueeze(0).to(DEV); tok = torch.tensor([list(data)], dtype=torch.long, device=DEV)
        mask = torch.ones(1, len(data), device=DEV); _ = MODEL(feats, tok, mask)
        H = _acts[LAYER][0].float().cpu().numpy()
    P = ((H - CENTER[None, :]) @ COMPS.T) * SCALE
    out = []
    for t, b in enumerate(data):
        ch = chr(b) if 32 <= b < 127 else (chr(b) if b >= 128 else "·")
        out.append({"x": float(P[t, 0]), "y": float(P[t, 1]), "z": float(P[t, 2]), "ch": ch, "b": int(b)})
    return out


@torch.no_grad()
def _asym_generate(seed: bytes, n_new=80, temp=0.8):
    """AsymHSL generation: seed = DENSE input context (encoder); AR-decode the continuation."""
    from hsl_asym import pack_input, out_features
    inpf = pack_input(seed, KK).unsqueeze(0).to(DEV)              # input memory = seed
    out = bytearray(b" ")                                         # prime output with a space
    for _ in range(n_new):
        of = out_features(bytes(out)).unsqueeze(0).to(DEV)
        logits = MODEL(inpf, of)[0, -1]
        nxt = int(logits.argmax()) if temp <= 0 else int(torch.multinomial((logits/temp).softmax(-1), 1))
        if nxt >= 256: break
        out.append(nxt)
    return bytes(out)                                             # leading space + generated


@torch.no_grad()
def trajectory(text: str, mode="read", max_bytes=400):
    seed = (text.encode("utf-8", "replace")[:max_bytes]) or b" "
    if mode == "generate":
        if ARCH == "asym":
            gen = _asym_generate(bytes(seed), n_new=80)
            full = (bytes(seed) + gen)[:max_bytes + 100]
            return {"points": _read_points(full), "text": full.decode("utf-8", "replace"),
                    "gen_start": len(seed), "mode": "generate"}
        full = MODEL.generate(bytes(seed), n_new=80, window=256, device=DEV, temperature=0.8)
        data = full[:max_bytes + 100]
        return {"points": _read_points(data), "text": data.decode("utf-8", "replace"),
                "gen_start": len(seed), "mode": "generate"}
    return {"points": _read_points(seed), "text": seed.decode("utf-8", "replace"), "gen_start": 0, "mode": "read"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/assets"):
            self._send(200, json.dumps({"stars": ASSETS["stars"], "meta": ASSETS["meta"]}))
        else:
            out_page = PAGE
            if "embed=true" in self.path:
                out_page = out_page.replace("<style>", "<style>#barWrap, #cap, #hint { display: none !important; }</style><style>")
            self._send(200, out_page, "text/html; charset=utf-8")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n).decode("utf-8", "replace")
        try:
            req = json.loads(body);
            self._send(200, json.dumps(trajectory(req.get("text", ""), req.get("mode", "read"))))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


PAGE = r"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8><title>byte universe — trajectory</title>
<style>
 html,body{margin:0;height:100%;background:#05060a;overflow:hidden;font-family:ui-monospace,monospace;color:#cdd6f4}
 #cap, #hint { display: none !important; }
</style></head><body>
<div id=cap></div><div id=hint></div>
<script type="importmap">{"imports":{
 "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const scene=new THREE.Scene(); scene.fog=new THREE.FogExp2(0x05060a,0.0015);
const cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,4000); cam.position.set(160,90,200);
const rdr=new THREE.WebGLRenderer({antialias:true}); rdr.setSize(innerWidth,innerHeight); rdr.setPixelRatio(Math.min(devicePixelRatio,2));
document.body.appendChild(rdr.domElement);
const ctrl=new OrbitControls(cam,rdr.domElement); ctrl.enableDamping=true; ctrl.autoRotate=true; ctrl.autoRotateSpeed=0.35;
function glow(){const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d');
 const gr=g.createRadialGradient(32,32,0,32,32,32);gr.addColorStop(0,'#fff');gr.addColorStop(.25,'rgba(255,255,255,.85)');
 gr.addColorStop(.5,'rgba(255,255,255,.35)');gr.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=gr;g.fillRect(0,0,64,64);
 return new THREE.CanvasTexture(c);}
const TEX=glow();
const sg=new THREE.BufferGeometry(),sv=[];for(let i=0;i<1200;i++)sv.push((Math.random()-.5)*900,(Math.random()-.5)*900,(Math.random()-.5)*900);
sg.setAttribute('position',new THREE.Float32BufferAttribute(sv,3));
scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0x223044,size:1.1})));
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rdr.setSize(innerWidth,innerHeight)});

let stars=[];
fetch('/assets').then(r=>r.json()).then(d=>{
 for(const p of d.stars){const m=new THREE.SpriteMaterial({map:TEX,color:new THREE.Color(p.color),
   blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:.9});
   const s=new THREE.Sprite(m);const sz=4+p.f*22;s.scale.set(sz,sz,1);s.position.set(p.x,p.y,p.z);scene.add(s);stars.push(s);}
});

const comet=new THREE.Sprite(new THREE.SpriteMaterial({map:TEX,color:0xffffff,blending:THREE.AdditiveBlending,depthWrite:false}));
comet.scale.set(14,14,1); comet.visible=false; scene.add(comet);
let trailGeo=new THREE.BufferGeometry(), trail=new THREE.Line(trailGeo,new THREE.LineBasicMaterial({color:0x9ad0ff,transparent:true,opacity:.6,blending:THREE.AdditiveBlending}));
scene.add(trail);
let path=[], pi=0, sub=0, playing=false, genStart=0, curMode='read';
const SEED=new THREE.Color(0x9ad0ff), GEN=new THREE.Color(0xffd86b);

function fire(text, mode){
 if (!text || text === "안녕하세요, 만물은 0과 1의 요동이다.") return;
 playing=false; pi=0; sub=0; path=[]; visited.length=0;
 curMode=mode || 'generate';
 fetch('/traj',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode:curMode})})
  .then(r=>r.json()).then(d=>{
    if(d.error) return;
    path=d.points; pi=0; sub=0; playing=true; comet.visible=true; visited.length=0;
    genStart=d.gen_start||0; curMode=d.mode||mode;
  }).catch(e=>console.error(e));
}

const visited=[];
function step(){
 if(!playing||path.length<2)return;
 sub+=0.25;
 if(sub>=1){sub=0;pi++;}
 if(pi>=path.length-1){playing=false;return;}
 const a=path[pi],b=path[pi+1];
 const x=a.x+(b.x-a.x)*sub,y=a.y+(b.y-a.y)*sub,z=a.z+(b.z-a.z)*sub;
 comet.position.set(x,y,z);
 const inGen = (curMode==='generate' && pi>=genStart);
 trail.material.color.copy(inGen?GEN:SEED);
 visited.push(x,y,z); if(visited.length>3*240)visited.splice(0,3);
 trailGeo.setAttribute('position',new THREE.Float32BufferAttribute(visited.slice(),3));
 trailGeo.setDrawRange(0,visited.length/3);
}
(function loop(){requestAnimationFrame(loop);ctrl.update();step();rdr.render(scene,cam);})();

const urlParams = new URLSearchParams(window.location.search);
const txt = urlParams.get('text');
const md = urlParams.get('mode') || 'generate';
if (txt && urlParams.get('autorun') === '1') {
    fire(txt, md);
}
</script></body></html>"""


if __name__ == "__main__":
    port = 8778
    print(f"trajectory server -> http://127.0.0.1:{port}  (type text, watch the comet)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

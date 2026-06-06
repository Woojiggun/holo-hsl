"""Render the model's self-built byte universe as an interactive 3D scene (three.js).

Reads a trained checkpoint, takes the per-byte mean representation at a chosen layer,
projects to 3D (PCA), and emits a SELF-CONTAINED .html: each byte = a glowing star,
size = frequency (hub-ness), color = byte category, orbit-controls + autorotate, dark space.
No matplotlib — a real WebGL universe you rotate in the browser. "세계가 열렸다" made literal.
"""
from __future__ import annotations
import sys, pathlib, json, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, torch
from hsl_signal_encoder import signal_features, FEAT_DIM
from hsl_decoder import HSLDecoder, EOS_ID
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POC_VAL = r"C:\Users\gguni\holo_v6_data\val.jsonl"

CATS = {  # category : (label, hex color)
    "letter": ("ascii letter", "#4da6ff"), "digit": ("digit", "#b070ff"),
    "punct": ("punct/space", "#2ecc71"), "high": ("UTF-8 / 한글 등", "#ff5050"),
    "space": ("space", "#ffffff"), "newline": ("newline", "#7f7f7f"), "control": ("control", "#d4d44a"),
}
def cat_of(b):
    if b >= 128: return "high"
    if b == 32: return "space"
    if b == 10: return "newline"
    c = chr(b)
    if c.isalpha(): return "letter"
    if c.isdigit(): return "digit"
    if b < 32: return "control"
    return "punct"


def load_stream(path, max_bytes):
    buf = bytearray()
    for line in open(path, encoding="utf-8", errors="replace"):
        try: t = json.loads(line).get("text") or ""
        except json.JSONDecodeError: continue
        buf.extend(t.encode("utf-8", "replace")); buf.append(EOS_ID & 0xFF)
        if len(buf) >= max_bytes: break
    return np.frombuffer(bytes(buf[:max_bytes]), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "hsl_lm_main_512" / "checkpoint_step_020000.pt"))
    ap.add_argument("--layer", type=int, default=6, help="transformer layer to read the universe from (-1=embedding)")
    ap.add_argument("--max-bytes", type=int, default=200_000)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", default=str(HERE / "universe_3d.html"))
    ap.add_argument("--arch", choices=["decoder", "asym"], default="decoder")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEV); cfg = ck.get("config", {})
    dim, heads = cfg.get("dim", 512), cfg.get("heads", 8)
    if args.arch == "asym":
        from hsl_asym import AsymHSL, pack_input, out_features
        layers = cfg.get("dec_layers", 12); K = cfg.get("K", 8)
        model = AsymHSL(dim=dim, enc_layers=cfg.get("enc_layers", 4), dec_layers=layers, heads=heads, K=K).to(DEV).eval()
        blocks = model.self_blocks
    else:
        layers = cfg.get("layers", 12); K = None
        model = HSLDecoder(dim, layers, heads).to(DEV).eval()
        blocks = model.blocks
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    print(f"loaded {pathlib.Path(args.ckpt).name} arch={args.arch} dim{dim}/L{layers}  reading layer {args.layer}")

    acts = {}
    for i, blk in enumerate(blocks):
        blk.register_forward_hook((lambda i: (lambda _m, _i, o: acts.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())))(i))

    by = load_stream(POC_VAL, args.max_bytes); L = args.seq; n = len(by) // L
    ssum = np.zeros((256, dim), np.float64); cnt = np.zeros(256, np.int64)
    with torch.no_grad():
        for c in range(n):
            seg = bytes(by[c*L:(c+1)*L]); f, _ = signal_features(seg)
            feats = f.unsqueeze(0).to(DEV)
            if args.arch == "asym":
                inpf = pack_input(seg, K).unsqueeze(0).to(DEV)
                emb = model.out_proj(feats); _ = model(inpf, feats)
            else:
                tok = torch.tensor([list(seg)], dtype=torch.long, device=DEV); mask = torch.ones(1, L, device=DEV)
                emb = model.embed(feats, tok); _ = model(feats, tok, mask)
            rep = (emb if args.layer < 0 else acts[args.layer])[0].float().cpu().numpy()
            ids = np.frombuffer(seg, np.uint8); np.add.at(ssum, ids, rep); np.add.at(cnt, ids, 1)

    seen = np.where(cnt > 0)[0]
    mean = ssum[seen] / cnt[seen][:, None]
    center = mean.mean(0)                                       # basis center (subtract from any new state)
    X = mean - center[None, :]
    # PCA -> 3D
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:3]                                              # [3, dim] projection basis
    P0 = X @ comps.T
    scale = 100.0 / (np.abs(P0).max() + 1e-9)                   # scale to ~[-100,100]
    P = P0 * scale
    freq = cnt[seen] / cnt[seen].sum()
    logf = np.log(freq); logf = (logf - logf.min()) / (logf.max() - logf.min() + 1e-9)

    pts = []
    for k, b in enumerate(seen.tolist()):
        cat = cat_of(b)
        ch = chr(b) if 32 <= b < 127 else (f"0x{b:02X}")
        pts.append({"b": int(b), "ch": ch, "x": float(P[k, 0]), "y": float(P[k, 1]), "z": float(P[k, 2]),
                    "f": float(logf[k]), "cat": cat, "color": CATS[cat][1]})
    # hub correlation for the caption
    radius = np.linalg.norm(P, axis=1)
    hub = float(np.corrcoef(np.log(freq), radius)[0, 1])
    meta = {"ckpt": pathlib.Path(args.ckpt).name, "layer": args.layer, "step": ck.get("step", "?"),
            "hub_corr": round(hub, 3), "n_bytes": int(len(seen)),
            "legend": [{"label": v[0], "color": v[1]} for v in CATS.values()]}
    html = HTML_TEMPLATE.replace("/*DATA*/", json.dumps(pts)).replace("/*META*/", json.dumps(meta))
    pathlib.Path(args.out).write_text(html, encoding="utf-8")
    # assets for the interactive trajectory server (stars + projection basis)
    assets = {"stars": pts, "meta": meta,
              "basis": {"center": center.tolist(), "comps": comps.tolist(), "scale": float(scale),
                        "layer": args.layer, "ckpt": str(args.ckpt), "arch": args.arch,
                        "dim": dim, "layers": layers, "heads": heads}}
    pathlib.Path(HERE / "universe_assets.json").write_text(json.dumps(assets), encoding="utf-8")
    print(f"hub corr(logfreq, radius) = {hub:+.3f} | {len(seen)} bytes")
    print(f"wrote -> {args.out}  (open in a browser, drag to orbit)")
    print(f"wrote -> universe_assets.json  (stars + projection basis for the trajectory server)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>HSL — self-built byte universe</title>
<style>
  html,body{margin:0;height:100%;background:#05060a;overflow:hidden;font-family:ui-monospace,Menlo,monospace;color:#cdd6f4}
  #hud{position:fixed;top:14px;left:16px;z-index:10;font-size:12px;line-height:1.5;text-shadow:0 0 8px #000}
  #hud h1{font-size:15px;margin:0 0 6px;font-weight:600;letter-spacing:.5px}
  #hud .dim{color:#7f849c}
  #legend{position:fixed;bottom:14px;left:16px;z-index:10;font-size:11px;line-height:1.6}
  #legend .sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle;box-shadow:0 0 6px currentColor}
  #tip{position:fixed;z-index:11;pointer-events:none;background:rgba(10,12,20,.85);border:1px solid #313244;border-radius:6px;
       padding:4px 8px;font-size:12px;display:none;text-shadow:none}
  #hint{position:fixed;bottom:14px;right:16px;z-index:10;font-size:11px;color:#7f849c}
</style></head><body>
<div id="hud"><h1>SELF-BUILT BYTE UNIVERSE</h1>
  <div class="dim" id="sub"></div></div>
<div id="legend"></div>
<div id="tip"></div>
<div id="hint">drag = orbit · scroll = zoom · auto-rotating</div>
<script type="importmap">{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"
}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const PTS=/*DATA*/, META=/*META*/;
document.getElementById('sub').innerHTML =
  `${META.n_bytes} byte-stars · layer ${META.layer} · step ${META.step}<br>`+
  `hub corr(freq,radius) = <b>${META.hub_corr}</b> &nbsp;<span class="dim">(model's own grid, non-human)</span>`;
const leg=document.getElementById('legend');
META.legend.forEach(l=>{leg.innerHTML+=`<div><span class="sw" style="color:${l.color}"></span>${l.label}</div>`});

const scene=new THREE.Scene();
scene.fog=new THREE.FogExp2(0x05060a,0.0016);
const cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,4000);
cam.position.set(160,90,200);
const rdr=new THREE.WebGLRenderer({antialias:true});
rdr.setSize(innerWidth,innerHeight);
rdr.setPixelRatio(Math.min(devicePixelRatio,2));
document.body.appendChild(rdr.domElement);
const ctrl=new OrbitControls(cam,rdr.domElement);
ctrl.enableDamping=true; ctrl.autoRotate=false;   // galaxy does its own differential spin

// glow sprite texture (radial gradient on canvas)
function glowTex(){const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d');
  const gr=g.createRadialGradient(32,32,0,32,32,32);
  gr.addColorStop(0,'rgba(255,255,255,1)');gr.addColorStop(0.25,'rgba(255,255,255,.85)');
  gr.addColorStop(0.5,'rgba(255,255,255,.35)');gr.addColorStop(1,'rgba(255,255,255,0)');
  g.fillStyle=gr;g.fillRect(0,0,64,64);return new THREE.CanvasTexture(c);}
const TEX=glowTex();

// background starfield
const sg=new THREE.BufferGeometry();const sv=[];
for(let i=0;i<1400;i++){const r=900;sv.push((Math.random()-.5)*r,(Math.random()-.5)*r,(Math.random()-.5)*r);}
sg.setAttribute('position',new THREE.Float32BufferAttribute(sv,3));
scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0x223044,size:1.1,sizeAttenuation:true})));

// center hub marker
const hub=new THREE.Mesh(new THREE.SphereGeometry(2.2,24,24),
  new THREE.MeshBasicMaterial({color:0xffffff}));scene.add(hub);
const halo=new THREE.Sprite(new THREE.SpriteMaterial({map:TEX,color:0x88aaff,blending:THREE.AdditiveBlending,depthWrite:false}));
halo.scale.set(26,26,1);scene.add(halo);

// byte stars (these will orbit the hub)
const sprites=[]; const Y=new THREE.Vector3(0,1,0);
for(const p of PTS){
  const m=new THREE.SpriteMaterial({map:TEX,color:new THREE.Color(p.color),
    blending:THREE.AdditiveBlending,depthWrite:false,transparent:true});
  const s=new THREE.Sprite(m);
  const sz=4+p.f*22; s.scale.set(sz,sz,1);
  s.position.set(p.x,p.y,p.z); s.userData=p; scene.add(s); sprites.push(s);
}
// tiny satellite particles orbiting the brightest hubs (자전 느낌)
const sats=[]; const hubs=[...sprites].sort((a,b)=>b.userData.f-a.userData.f).slice(0,14);
for(const h of hubs){
  const nsat=2+Math.floor(h.userData.f*3);
  for(let i=0;i<nsat;i++){
    const sm=new THREE.SpriteMaterial({map:TEX,color:new THREE.Color(h.userData.color),
      blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:0.9});
    const sp=new THREE.Sprite(sm); const sc=1.3+Math.random()*1.7; sp.scale.set(sc,sc,1); scene.add(sp);
    const n=new THREE.Vector3(Math.random()-.5,Math.random()-.5,Math.random()-.5).normalize();
    let u=new THREE.Vector3(1,0,0); if(Math.abs(n.x)>0.9)u.set(0,1,0);
    u.sub(n.clone().multiplyScalar(u.dot(n))).normalize();
    const v=new THREE.Vector3().crossVectors(n,u);
    sats.push({mesh:sp,parent:h,u,v,r:h.scale.x*0.7+2+Math.random()*4,
      ang:Math.random()*6.283,spd:0.025+Math.random()*0.05});
  }
}

// hover tooltip via raycaster
const ray=new THREE.Raycaster(); ray.params.Sprite={}; const mouse=new THREE.Vector2();
const tip=document.getElementById('tip');
addEventListener('pointermove',e=>{
  mouse.x=(e.clientX/innerWidth)*2-1; mouse.y=-(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mouse,cam); const hit=ray.intersectObjects(sprites,false);
  if(hit.length){const d=hit[0].object.userData;
    tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
    tip.innerHTML=`<b>'${d.ch}'</b> &nbsp;byte ${d.b} &nbsp;<span style="color:${d.color}">●</span> ${d.cat}`;
  } else tip.style.display='none';
});
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rdr.setSize(innerWidth,innerHeight);});
const _tmp=new THREE.Vector3();
(function loop(){requestAnimationFrame(loop);ctrl.update();halo.material.rotation+=0.002;
  for(const s of sprites){const R=Math.hypot(s.position.x,s.position.z);
    s.position.applyAxisAngle(Y, 0.0040/(R*0.011+1));}        // differential orbit: inner hubs faster
  for(const st of sats){st.ang+=st.spd;
    _tmp.copy(st.u).multiplyScalar(Math.cos(st.ang)*st.r).addScaledVector(st.v,Math.sin(st.ang)*st.r);
    st.mesh.position.copy(st.parent.position).add(_tmp);}      // particles ride their parent hub
  rdr.render(scene,cam);})();
</script></body></html>"""


if __name__ == "__main__":
    main()

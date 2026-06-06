"""Max bytes per INPUT embedding (for the asymmetric dense-input/byte-output design).
Patch AE at D=512, trained LONG (training raises recon), high K. Find the K where the
input embedding still faithfully holds the bytes. recon at step 2000 vs 12000 shows the
training effect. (Recon = conservative proxy; a cross-attn decoder tolerates more loss.)
"""
from __future__ import annotations
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POC = r"G:\내 드라이브\홀로빗 POC\data\train.jsonl"
D = 512

def load_stream(n):
    buf = bytearray()
    with open(POC, encoding="utf-8", errors="replace") as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            t = d.get("text") or d.get("serialized") or d.get("output") or d.get("input") or ""
            buf.extend(t.encode("utf-8","replace"))
            if len(buf) >= n: break
    return bytes(buf[:n])

class AE(nn.Module):
    def __init__(self, K, e=48, h=1024):
        super().__init__(); self.K=K; self.emb=nn.Embedding(256,e)
        self.enc=nn.Sequential(nn.Linear(K*e,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,D))
        self.dec=nn.Sequential(nn.Linear(D,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,K*256))
    def forward(self, ids):
        return self.dec(self.enc(self.emb(ids).reshape(ids.shape[0],-1))).reshape(-1,self.K,256)

def patches(stream, K, n):
    npatch=min(n, len(stream)//K)
    ids=torch.zeros(npatch,K,dtype=torch.long)
    for i in range(npatch): ids[i]=torch.tensor(list(stream[i*K:(i+1)*K]))
    return ids

def run(stream, K, checkpoints=(2000,12000), bs=128):
    ids=patches(stream[:(len(stream)//K)*K], K, 40000)
    ntr=int(ids.shape[0]*0.85); itr,iva=ids[:ntr].to(DEV),ids[ntr:].to(DEV)
    torch.manual_seed(0); m=AE(K).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    out={}; step=0
    for tgt in checkpoints:
        for _ in range(tgt-step):
            idx=torch.randint(0,itr.shape[0],(bs,),device=DEV)
            loss=F.cross_entropy(m(itr[idx]).reshape(-1,256),itr[idx].reshape(-1))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        step=tgt; m.eval()
        with torch.no_grad(): out[tgt]=float((m(iva).argmax(-1)==iva).float().mean())
        m.train()
    return out

def main():
    stream=load_stream(8_000_000)
    print(f"D={D} (our model dim) | recon = bytes faithfully held in ONE input embedding | {DEV}\n")
    print(f"{'K(bytes/emb)':>12} {'recon@2k':>9} {'recon@12k':>10} {'gain':>7}")
    for K in [8,16,32,64,128,256]:
        c=run(stream,K)
        print(f"{K:>12} {c[2000]:>9.3f} {c[12000]:>10.3f} {c[12000]-c[2000]:>+7.3f}", flush=True)
    print("\nhigh recon@12k = faithfully readable input at that K. find the max K still ~>=0.9.")
    print("(asymmetric model: input context = positions x K bytes; cross-attn decoder tolerates more loss.)")

if __name__=="__main__":
    main()

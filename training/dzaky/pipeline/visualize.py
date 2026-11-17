"""MS-TCN stage -- watch the model work: one self-contained HTML player per window of a match.

    python -m pipeline.visualize                                  # BAR-ATH, 5 min from kickoff
    python -m pipeline.visualize --start 40:00 --minutes 3        # another window (video clock)
    python -m pipeline.visualize --truth evaluation/gt/<public gt>.csv   # add the public labels
    python -m pipeline.visualize --dfl J03WR9                     # a DFL match instead

The page shows, frame by frame and in sync:

    a top-down pitch: both teams, the ball, and where it was last seen while it is missing
    the model's probabilities H / A / D over time, and the labels they decode to
    the rule-based labels, the ball-visible track, and the truth when there is one

No video for BAR-ATH: the organisers' clarification of 2026-11-04 forbids checking output against
the provided video, so the page never embeds it (inspecting the video on its own, to understand
what counts as possession, is allowed -- outside this tool).

BAR-ATH runs only on your machine: the page embeds the tracking of the window, so keep it local
(evaluation/visualizations/ is git-ignored). Nothing is printed but the output path.

For a DFL match the page uses the fold model that never saw that match, the challenge-style
labels (pipeline.dfl_events) as truth, and the ATD-fitted damage, so it shows the model as it
behaves on data it was not trained on.
"""

import argparse
import base64
import json
import os
import sys

import numpy as np

from pipeline import features
from pipeline.config import ROOT, Config
from pipeline.decode import combine, decode, flipped_probabilities
from pipeline.labels import label_match
from pipeline.run_tcn import DECODE_MIN_S, DECODE_SWITCH, DEFAULT_DEAD_MODEL, DEFAULT_MODEL
from pipeline.tcn import Predictor, deterministic

OUT_DIR = os.path.join(ROOT, "evaluation", "visualizations")
MISSING = -32768


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _metres(values):
    """Positions as int16 decimetres, MISSING where NaN."""
    scaled = np.clip(np.round(np.nan_to_num(values) * 10), -32767, 32767)
    return np.where(np.isnan(values), MISSING, scaled).astype("<i2")


def model_view(tracking, halves, predictor, cfg, dead=None):
    """(probabilities, model labels, rule labels) for the whole match, as run_tcn makes them.

    dead: a second Predictor whose dead ball is used (decode.combine), or None.
    """
    cfg = cfg.replace(PAKAI_SMART_TAGGING=False)
    rules = label_match(tracking, [], halves, cfg).labels
    table = features.compute(tracking, halves, cfg, rule_labels=rules)
    probabilities = flipped_probabilities(predictor, table, halves)
    if dead is not None:
        probabilities = combine(probabilities, flipped_probabilities(dead, table, halves))
    labels = decode(probabilities, halves, DECODE_SWITCH, DECODE_MIN_S, cfg.FPS)
    return probabilities, labels, rules


def build_view(tracking, halves, cfg, predictor, start, stop, truth=None, title="", video=None,
               note="", dead=None):
    """Everything the page needs for frames [start, stop), as a JSON-ready dict."""
    start, stop = max(0, int(start)), min(len(tracking), int(stop))
    probabilities, labels, rules = model_view(tracking, halves, predictor, cfg, dead)
    window = slice(start, stop)
    players = tracking.players[window]
    used = np.flatnonzero(~np.isnan(players[:, :, 0]).all(0))
    return {
        "title": title, "note": note, "fps": cfg.FPS, "start": int(start),
        "n": int(stop - start), "pitch": [cfg.LAPANGAN_P, cfg.LAPANGAN_L],
        "team": tracking.team[used].astype(int).tolist(), "slots": int(len(used)),
        "players": _b64(_metres(players[:, used])), "ball": _b64(_metres(tracking.ball[window])),
        "probs": _b64(np.round(probabilities[window] * 255).astype(np.uint8)),
        "model": "".join(labels[window]), "rules": "".join(rules[window]),
        "truth": "".join(truth[window]) if truth is not None else None,
        "video": video,
    }


def write_page(view, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = json.dumps(view, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(PAGE.replace("__TITLE__", view["title"]).replace("__DATA__", data))
    return path


def parse_clock(text, fps):
    """'mm:ss' or a frame number -> frame."""
    if ":" in text:
        minutes, seconds = text.split(":")
        return int(round((int(minutes) * 60 + float(seconds)) * fps))
    return int(text)


def bar_ath(args):
    sys.path.insert(0, ROOT)
    from games import GAME, half_video_blocks
    from pipeline.data import load_tracking

    cfg = Config()
    tracking = load_tracking(cfg=cfg)
    halves = half_video_blocks(GAME)
    start = parse_clock(args.start, cfg.FPS) if args.start else halves[0][1]
    stop = min(len(tracking), start + int(args.minutes * 60 * cfg.FPS))
    truth = None
    if args.truth:
        sys.path.insert(0, os.path.join(ROOT, "evaluation"))
        import possession_metrics as pm
        table = pm.read_label_csv(args.truth)
        frames = table["frame"].to_numpy(dtype=np.int64)
        keep = (frames >= 0) & (frames < len(tracking))
        truth = np.full(len(tracking), "X", dtype="<U1")
        truth[frames[keep]] = [str(v).strip().upper() for v in table["label"].to_numpy()[keep]]
    out = args.out or os.path.join(OUT_DIR, "{}_{}.html".format(GAME.lower(), start))
    dead = None if args.dead_model.lower() == "none" else Predictor.load(args.dead_model)
    name = os.path.basename(args.model)
    if dead is not None:
        name += " + dead ball from " + os.path.basename(args.dead_model)
    view = build_view(tracking, halves, cfg, Predictor.load(args.model), start, stop, truth,
                      title="{} - {}".format(GAME, name),
                      note="Models trained on Sportec Open DFL only.", dead=dead)
    return write_page(view, out)


def dfl(args):
    from pipeline.atd_noise import damage
    from pipeline.dfl_events import load_relabelled
    from pipeline.robustness import SCENARIOS
    from pipeline.train_tcn import fold_model_path

    tracking, truth, halves, cfg = load_relabelled(args.dfl)
    damaged = damage(tracking, truth, halves, SCENARIOS["atd_fit"][0], seed=0)
    start = parse_clock(args.start, cfg.FPS) + halves[0][1] if args.start else halves[0][1]
    stop = min(len(truth), start + int(args.minutes * 60 * cfg.FPS))
    model = fold_model_path(args.fold_name, args.dfl)
    dead = None
    name = os.path.basename(model)
    if args.dead_fold_name.lower() != "none":
        dead_path = fold_model_path(args.dead_fold_name, args.dfl)
        dead = Predictor.load(dead_path)
        name += " + dead ball from " + os.path.basename(dead_path)
    out = args.out or os.path.join(OUT_DIR, "dfl_{}_{}.html".format(args.dfl, start))
    view = build_view(damaged, halves, cfg, Predictor.load(model), start, stop, truth,
                      title="DFL {} - {} (held out)".format(args.dfl, name), dead=dead,
                      note="Sportec Open DFL (CC BY), damaged like the ATD; truth = "
                           "challenge-style labels. The model never saw this match.")
    return write_page(view, out)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;
--H:#2563eb;--A:#e0582b;--D:#9a9a94;--miss:#e2b007;--pitch:#3f8a4f;--pline:rgba(255,255,255,.75)}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#21211f;--ink:#ecebe6;
--mute:#9d9c96;--line:#33332f;--pitch:#2c5e37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px}.note{color:var(--mute);margin:0 0 12px}
.top{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.top.solo{grid-template-columns:minmax(0,760px)}
@media (max-width:820px){.top{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
video,canvas{display:block;width:100%}video{border-radius:6px;background:#000}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
button,select{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);
background:var(--panel);color:var(--ink);cursor:pointer}
.read{display:flex;flex-wrap:wrap;gap:16px;font-variant-numeric:tabular-nums}
.read b{display:inline-block;min-width:20px;text-align:center;border-radius:4px;color:#fff;padding:0 6px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--mute);font-size:13px;margin-top:8px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.lab{color:var(--mute);font-size:12px;margin:8px 0 4px}
</style></head><body><main>
<h1 id="title"></h1><p class="note" id="note"></p>
<div class="top">
 <div class="card" id="videoCard"><video id="video" muted playsinline preload="auto"></video></div>
 <div class="card"><canvas id="pitch"></canvas></div>
</div>
<div class="bar">
 <button id="play">Play</button>
 <select id="rate"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>
 <div class="read" id="read"></div>
</div>
<div class="card">
 <div class="lab">Around the playhead (&plusmn;30 s) &mdash; click to jump</div><canvas id="detail"></canvas>
 <div class="lab">Whole window</div><canvas id="overview"></canvas>
 <div class="legend">
  <span><i class="sw" style="background:var(--H)"></i>Home (H)</span>
  <span><i class="sw" style="background:var(--A)"></i>Away (A)</span>
  <span><i class="sw" style="background:var(--D)"></i>Dead ball (D)</span>
  <span><i class="sw" style="background:var(--miss)"></i>Ball not seen</span>
  <span>Top band: the model's probabilities. Rows below: the labels they decode to, the rule-based labels, the truth, and when the ball is missing.</span>
 </div>
</div>
</main>
<script>
const V=__DATA__;
const css=getComputedStyle(document.documentElement),C=k=>css.getPropertyValue(k).trim();
const COL={H:C('--H'),A:C('--A'),D:C('--D'),X:'transparent'};
function dec(b,T){const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer)}
const P=dec(V.players,Int16Array),B=dec(V.ball,Int16Array),PR=dec(V.probs,Uint8Array),S=V.slots,N=V.n,F=V.fps;
document.getElementById('title').textContent=V.title;document.getElementById('note').textContent=V.note;
const rows=[['model',V.model],['rules',V.rules]];if(V.truth)rows.push(['truth',V.truth]);
const seen=new Uint8Array(N),last=new Int32Array(N);let l=-1;
for(let i=0;i<N;i++){seen[i]=B[2*i]!==-32768;if(seen[i])l=i;last[i]=l}
let frame=0,playing=false,clock=null;
const video=document.getElementById('video');
if(V.video){video.src=V.video;video.addEventListener('loadedmetadata',()=>seek(0))}else{document.getElementById('videoCard').style.display='none';document.querySelector('.top').classList.add('solo')}
function fit(c,h){const r=devicePixelRatio||1,w=c.clientWidth;c.width=w*r;c.height=h*r;c.style.height=h+'px';const x=c.getContext('2d');x.setTransform(r,0,0,r,0,0);return [x,w,h]}
function pitch(){
 const c=document.getElementById('pitch'),[L,W]=V.pitch,w=c.clientWidth,h=w*W/L;
 const [x]=fit(c,h),k=w/L;x.fillStyle=C('--pitch');x.fillRect(0,0,w,h);
 x.strokeStyle=C('--pline');x.lineWidth=1.2;const R=(a,b,c2,d)=>x.strokeRect(a*k,b*k,c2*k,d*k);
 R(0,0,L,W);x.beginPath();x.moveTo(L/2*k,0);x.lineTo(L/2*k,h);x.stroke();
 x.beginPath();x.arc(L/2*k,W/2*k,9.15*k,0,7);x.stroke();
 R(0,W/2-20.16,16.5,40.32);R(L-16.5,W/2-20.16,16.5,40.32);R(0,W/2-9.16,5.5,18.32);R(L-5.5,W/2-9.16,5.5,18.32);
 for(let s=0;s<S;s++){const j=(frame*S+s)*2;if(P[j]===-32768)continue;
  x.fillStyle=V.team[s]===0?COL.H:COL.A;x.beginPath();x.arc(P[j]/10*k,P[j+1]/10*k,4.5,0,7);x.fill();
  x.strokeStyle='rgba(255,255,255,.9)';x.lineWidth=1;x.stroke()}
 if(seen[frame]){x.fillStyle='#fff';x.strokeStyle='#111';x.lineWidth=1.5;x.beginPath();
  x.arc(B[2*frame]/10*k,B[2*frame+1]/10*k,4,0,7);x.fill();x.stroke()}
 else if(last[frame]>=0){const q=last[frame];x.setLineDash([3,3]);x.strokeStyle='#fff';x.lineWidth=1.5;
  x.beginPath();x.arc(B[2*q]/10*k,B[2*q+1]/10*k,6,0,7);x.stroke();x.setLineDash([])}
}
function band(x,w,top,h,a,b){ // probabilities, frames a..b over width w
 const step=(b-a)/w;for(let px=0;px<w;px++){const f0=Math.floor(a+px*step),f1=Math.max(f0+1,Math.floor(a+(px+1)*step));
  let s=[0,0,0],n=0;for(let f=f0;f<f1&&f<N;f++){if(f<0)continue;for(let c=0;c<3;c++)s[c]+=PR[3*f+c];n++}
  if(!n)continue;let y=top;['H','A','D'].forEach((L,c)=>{const hh=s[c]/n/255*h;x.fillStyle=COL[L];x.fillRect(px,y,1.02,hh);y+=hh})}
}
const GUT=48;
function strips(x,w,top,a,b,rh){
 const step=(b-a)/w;[...rows,['ball',null]].forEach(([name,str],r)=>{
  const y=top+r*(rh+3);x.fillStyle=C('--mute');x.font='11px system-ui';x.textBaseline='middle';x.fillText(name,0,y+rh/2);
  x.fillStyle='rgba(127,127,127,.12)';x.fillRect(GUT,y,w,rh);
  for(let px=0;px<w;px++){const f=Math.floor(a+(px+.5)*step);if(f<0||f>=N)continue;
   const col=str===null?(seen[f]?null:C('--miss')):COL[str[f]];if(!col)continue;x.fillStyle=col;x.fillRect(GUT+px,y,1.02,rh)}});
}
function timeline(c,h,a,b,cursor){
 const [x,W]=fit(c,h),w=W-GUT,rh=10,bandH=h-(rows.length+1)*(rh+3)-2;x.clearRect(0,0,W,h);
 x.fillStyle=C('--mute');x.font='11px system-ui';x.textBaseline='middle';x.fillText('P(H/A/D)',0,bandH/2);
 x.save();x.translate(GUT,0);band(x,w,0,bandH,a,b);x.restore();strips(x,w,bandH+4,a,b,rh);
 if(cursor){const px=GUT+(frame-a)/(b-a)*w;x.fillStyle=C('--ink');x.fillRect(px-1,0,2,h)}
 c.onclick=e=>{const r=c.getBoundingClientRect(),u=(e.clientX-r.left-GUT)/(r.width-GUT);if(u>=0)seek(Math.round(a+u*(b-a)))}
}
let over=null;
function overview(){const c=document.getElementById('overview');timeline(c,70,0,N,false);over=c.getContext('2d').getImageData(0,0,c.width,c.height)}
function drawOver(){const c=document.getElementById('overview'),x=c.getContext('2d');x.putImageData(over,0,0);
 const px=GUT+frame/N*(c.clientWidth-GUT);x.fillStyle=C('--ink');x.fillRect(px-1,0,2,c.clientHeight)}
function read(){
 const t=(V.start+frame)/F,mm=Math.floor(t/60),ss=(t%60).toFixed(1).padStart(4,'0');
 const p=[0,1,2].map(c=>(PR[3*frame+c]/255*100).toFixed(0)+'%');
 const tag=(n,L)=>`${n} <b style="background:${COL[L]==='transparent'?'#999':COL[L]}">${L}</b>`;
 const gap=seen[frame]?'visible':`missing ${((frame-Math.max(last[frame],0))/F).toFixed(1)} s`;
 document.getElementById('read').innerHTML=`<span>${mm}:${ss} &middot; frame ${V.start+frame}</span><span>P(H) ${p[0]} &middot; P(A) ${p[1]} &middot; P(D) ${p[2]}</span>`+
  rows.map(r=>`<span>${tag(r[0],r[1][frame])}</span>`).join('')+`<span>ball ${gap}</span>`;
}
function draw(){pitch();timeline(document.getElementById('detail'),110,frame-30*F,frame+30*F,true);drawOver();read()}
function seek(f){frame=Math.max(0,Math.min(N-1,f));if(V.video&&video.readyState)video.currentTime=(V.start+frame)/F+0.001;draw()}
function tick(now){if(!playing)return;
 if(V.video){frame=Math.max(0,Math.min(N-1,Math.round(video.currentTime*F)-V.start))}
 else{if(clock===null)clock=now;frame=Math.min(N-1,frame+Math.floor((now-clock)/1000*F*+rate.value));if((now-clock)/1000*F*+rate.value>=1)clock=now}
 draw();if(frame>=N-1){toggle();return}requestAnimationFrame(tick)}
const btn=document.getElementById('play'),rate=document.getElementById('rate');
function toggle(){playing=!playing;btn.textContent=playing?'Pause':'Play';clock=null;
 if(V.video){video.playbackRate=+rate.value;playing?video.play():video.pause()}if(playing)requestAnimationFrame(tick)}
btn.onclick=toggle;rate.onchange=()=>{video.playbackRate=+rate.value};
addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();toggle()}if(e.key==='ArrowRight')seek(frame+(e.shiftKey?F:1));if(e.key==='ArrowLeft')seek(frame-(e.shiftKey?F:1))});
addEventListener('resize',()=>{overview();draw()});overview();draw();
</script></body></html>
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="BAR-ATH: trained model (.pt)")
    parser.add_argument("--dead-model", default=DEFAULT_DEAD_MODEL,
                        help="BAR-ATH: model whose dead ball is used (.pt), or 'none'")
    parser.add_argument("--start", help="window start: mm:ss on the video clock, or a frame")
    parser.add_argument("--minutes", type=float, default=5.0, help="window length")
    parser.add_argument("--truth", help="BAR-ATH: a label CSV to show as truth (e.g. public GT)")
    parser.add_argument("--dfl", help="a DFL match id instead of BAR-ATH, e.g. J03WR9")
    parser.add_argument("--fold-name", default="tcn6", help="DFL: run name of the fold models")
    parser.add_argument("--dead-fold-name", default="tcn2",
                        help="DFL: run name of the dead-ball fold models, or 'none'")
    parser.add_argument("--out", help="output .html")
    args = parser.parse_args()

    deterministic()
    path = dfl(args) if args.dfl else bar_ath(args)
    print("wrote {}".format(os.path.relpath(path, ROOT)))

"""Create a standalone browser replay from a recorded evaluation trajectory."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('run')
args=parser.parse_args()
run=Path(args.run)
frames=json.loads((run/'trajectory.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
payload=json.dumps(dict(frames=frames,dt=manifest['task']['simulation']['time_step_s'],radius=manifest['task']['robot_defaults']['footprint_radius_m']))
html="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Swarm simulation replay</title><style>
body{background:#101820;color:#edf2f7;font:16px system-ui;margin:24px}canvas{background:#192734;border:1px solid #425466;max-width:100%;height:auto}button,input{margin:10px}#status{white-space:pre-wrap}p{max-width:850px;color:#b9cad8}
</style><h1>Swarm simulation replay</h1>
<p>Recorded formation-and-hold evaluation. This is a replay of the Python simulator, not Gazebo. Desired center distance: 1 m. The current policy has not passed task validation.</p>
<button id="play">Play / Pause</button><button id="reset">Restart</button>
<label>Speed <select id="speed"><option value="1">1x</option><option value="4">4x</option><option value="10" selected>10x</option></select></label>
<input id="seek" type="range" min="0" value="0" style="width:50%">
<div id="status"></div><canvas width="1000" height="650" id="view"></canvas>
<script>const data=PAYLOAD;const frames=data.frames,canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),seek=document.getElementById('seek');seek.max=frames.length-1;
let index=0,playing=false,last=null;const colors=['#5ac8fa','#ffbd59','#d992ff'];
let xs=[],ys=[];frames.forEach(f=>f.forEach(p=>{xs.push(p[0]);ys.push(p[1])}));
const xmin=Math.min(...xs)-1,xmax=Math.max(...xs)+1,ymin=Math.min(...ys)-1,ymax=Math.max(...ys)+1;
const scale=Math.min(900/(xmax-xmin),550/(ymax-ymin));const X=x=>50+(x-xmin)*scale,Y=y=>600-(y-ymin)*scale;
function draw(){const k=Math.min(frames.length-1,Math.floor(index)),f=frames[k];ctx.clearRect(0,0,1000,650);ctx.strokeStyle='#314453';ctx.fillStyle='#9eb3c7';ctx.font='12px system-ui';
for(let x=Math.ceil(xmin);x<=xmax;x++){ctx.beginPath();ctx.moveTo(X(x),Y(ymin));ctx.lineTo(X(x),Y(ymax));ctx.stroke();ctx.fillText(x+' m',X(x),625)}
for(let y=Math.ceil(ymin);y<=ymax;y++){ctx.beginPath();ctx.moveTo(X(xmin),Y(y));ctx.lineTo(X(xmax),Y(y));ctx.stroke();ctx.fillText(y+' m',5,Y(y))}
let distances=[];for(let i=0;i<f.length;i++)for(let j=i+1;j<f.length;j++){let d=Math.hypot(f[i][0]-f[j][0],f[i][1]-f[j][1]);distances.push('R'+i+'–R'+j+': '+d.toFixed(2)+' m');ctx.strokeStyle=d<=2*data.radius?'#ff6060':'#71869a';ctx.beginPath();ctx.moveTo(X(f[i][0]),Y(f[i][1]));ctx.lineTo(X(f[j][0]),Y(f[j][1]));ctx.stroke()}
f.forEach((p,i)=>{ctx.strokeStyle=colors[i%3];ctx.globalAlpha=.4;ctx.beginPath();for(let t=0;t<=k;t++){let q=frames[t][i];if(t===0)ctx.moveTo(X(q[0]),Y(q[1]));else ctx.lineTo(X(q[0]),Y(q[1]))}ctx.stroke();ctx.globalAlpha=1;ctx.fillStyle=colors[i%3];ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),data.radius*scale,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#101820';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(X(p[0]),Y(p[1]));ctx.lineTo(X(p[0]+data.radius*Math.cos(p[2])),Y(p[1]+data.radius*Math.sin(p[2])));ctx.stroke();ctx.lineWidth=1;ctx.fillStyle='white';ctx.fillText('Robot '+i,X(p[0])+8,Y(p[1])-data.radius*scale-8)});
seek.value=k;document.getElementById('status').textContent='Time: '+((k+1)*data.dt).toFixed(1)+' s  |  '+distances.join('  |  ')}
document.getElementById('play').onclick=()=>playing=!playing;document.getElementById('reset').onclick=()=>{index=0;draw()};seek.oninput=()=>{index=Number(seek.value);playing=false;draw()};
function tick(t){if(last!==null&&playing){index+=Math.min((t-last)/1000,.1)*Number(document.getElementById('speed').value)/data.dt;if(index>=frames.length-1){index=frames.length-1;playing=false}draw()}last=t;requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></html>""".replace('PAYLOAD',payload)
output=run/'replay.html'
output.write_text(html)
print(output.resolve())

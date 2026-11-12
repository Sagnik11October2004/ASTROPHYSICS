'use strict';
/* Interactive browser adaptation of the single-vortex front tracking benchmark
   in Sagnik11October2004/Fluids/Front_Tracking.py.
   Prescribed divergence-free velocity and particle-traced interface only.
   This is NOT a free-surface Navier–Stokes / surface-tension solver. */
(() => {
  const canvas=document.getElementById('fluid-front-canvas');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  if(!ctx)return;
  const scrub=document.getElementById('fluid-time');
  const timeOut=document.getElementById('fluid-time-output');
  const phaseOut=document.getElementById('fluid-phase');
  const returnOut=document.getElementById('fluid-return-error');
  const play=document.getElementById('fluid-play');
  const reset=document.getElementById('fluid-reset');
  const step=document.getElementById('fluid-step');
  const speed=document.getElementById('fluid-speed');
  const vectors=document.getElementById('fluid-velocity');
  const markers=document.getElementById('fluid-markers');
  const originalNote=document.getElementById('fluid-live-note');
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const T=8, dt=.02, count=180, frames=Math.round(T/dt);
  const TAU=Math.PI*2, sin=Math.sin,cos=Math.cos, PI=Math.PI;
  const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
  const initial=Array.from({length:count},(_,k)=>{
    const theta=TAU*k/count;
    return [0.5+0.15*cos(theta),0.75+0.15*sin(theta)];
  });
  // Same streamfunction and velocity direction as the public Python source:
  // ψ=A(t) sin²(πx) sin²(πy), with A(t)=cos(πt/T).
  // u=-∂ψ/∂y; v=∂ψ/∂x.
  function vel(x,y,t){
    const A=cos(PI*t/T), sx=sin(PI*x),sy=sin(PI*y);
    return [
      -2*A*PI*sx*sx*sy*cos(PI*y),
       2*A*PI*sy*sy*sx*cos(PI*x)
    ];
  }
  // Three-stage explicit update matching the public front-tracking prototype's
  // stage coefficients, evaluated at the corresponding intermediate times.
  function advance(points,t,h){
    return points.map(p=>{
      const k1=vel(p[0],p[1],t);
      const k2=vel(p[0]+h*k1[0],p[1]+h*k1[1],t+h);
      const p2x=p[0]+h*(k1[0]+k2[0])*.25;
      const p2y=p[1]+h*(k1[1]+k2[1])*.25;
      const k3=vel(p2x,p2y,t+h*.5);
      return [
        p[0]+h*(k1[0]+k2[0]+4*k3[0])/6,
        p[1]+h*(k1[1]+k2[1]+4*k3[1])/6
      ];
    });
  }
  // Retain particle IDs for reproducible reversal tracking. The author's
  // SciPy spline point-redistribution stage is not reproduced in this port.
  const states=[initial];
  for(let n=0;n<frames;n++)states.push(advance(states[n],n*dt,dt));
  function contourArea(pts){
    let a=0;for(let i=0;i<pts.length;i++){
      const p=pts[i],q=pts[(i+1)%pts.length];a+=p[0]*q[1]-q[0]*p[1];
    }
    return Math.abs(a)*.5;
  }
  const a0=contourArea(initial);
  let t=0,playing=!reduced.matches,visible=true,raf=0,last=0,speedFactor=1;
  let width=760,height=460,dpr=1;
  function resize(){
    const rect=canvas.getBoundingClientRect();
    width=Math.max(360,rect.width||760);
    height=width*.60;
    dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw();
  }
  function interpolate(time){
    const v=clamp(time/dt,0,frames),n=Math.min(Math.floor(v),frames),f=v-n;
    if(n===frames||f===0)return states[n];
    const a=states[n],b=states[n+1];
    return a.map((p,i)=>[p[0]+(b[i][0]-p[0])*f,p[1]+(b[i][1]-p[1])*f]);
  }
  function view(){
    const pad=Math.max(28,Math.round(width*.045));
    const side=Math.min(width-pad*2,height-pad*2);
    const x0=(width-side)/2,y0=(height-side)/2;
    return {x:x=>x0+x*side,y:y=>y0+(1-y)*side,side};
  }
  function draw(){
    if(!width||!height)return;
    ctx.clearRect(0,0,width,height);
    const grad=ctx.createLinearGradient(0,0,width,height);
    grad.addColorStop(0,'#061c2d');grad.addColorStop(1,'#04111e');
    ctx.fillStyle=grad;ctx.fillRect(0,0,width,height);
    const m=view();
    ctx.save();
    ctx.strokeStyle='rgba(167,212,225,.13)';ctx.lineWidth=1;
    for(let i=0;i<=10;i++){
      const f=i/10;
      ctx.beginPath();ctx.moveTo(m.x(f),m.y(0));ctx.lineTo(m.x(f),m.y(1));ctx.stroke();
      ctx.beginPath();ctx.moveTo(m.x(0),m.y(f));ctx.lineTo(m.x(1),m.y(f));ctx.stroke();
    }
    ctx.strokeStyle='rgba(171,210,226,.5)';
    ctx.strokeRect(m.x(0),m.y(1),m.side,m.side);
    if(!vectors||vectors.checked){
      const arr=12;
      ctx.strokeStyle='rgba(118,198,218,.33)';
      ctx.fillStyle='rgba(118,198,218,.45)';
      ctx.lineWidth=1;
      for(let iy=1;iy<arr;iy++)for(let ix=1;ix<arr;ix++){
        const x=ix/arr,y=iy/arr,v=vel(x,y,t),mag=Math.hypot(...v);
        if(mag<.0001)continue;
        const scl=Math.min(.028,mag*.013),ux=v[0]/mag,uy=v[1]/mag;
        const sx=m.x(x),sy=m.y(y),ex=m.x(x+ux*scl),ey=m.y(y+uy*scl);
        ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ex,ey);ctx.stroke();
        const ang=Math.atan2(ey-sy,ex-sx),sz=3;
        ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-sz*cos(ang-.55),ey-sz*sin(ang-.55));
        ctx.lineTo(ex-sz*cos(ang+.55),ey-sz*sin(ang+.55));ctx.closePath();ctx.fill();
      }
    }
    // Initial interface, ghosted throughout deformation.
    ctx.beginPath();
    for(let i=0;i<initial.length;i++){
      const p=initial[i]; if(i===0)ctx.moveTo(m.x(p[0]),m.y(p[1]));
      else ctx.lineTo(m.x(p[0]),m.y(p[1]));
    }
    ctx.closePath();ctx.setLineDash([6,7]);ctx.strokeStyle='rgba(245,170,126,.66)';
    ctx.lineWidth=1.8;ctx.stroke();ctx.setLineDash([]);
    const pts=interpolate(t);
    ctx.beginPath();
    for(let i=0;i<pts.length;i++){
      const p=pts[i];if(i===0)ctx.moveTo(m.x(p[0]),m.y(p[1]));
      else ctx.lineTo(m.x(p[0]),m.y(p[1]));
    }
    ctx.closePath();ctx.fillStyle='rgba(66,195,213,.11)';ctx.fill();
    ctx.strokeStyle='#8ae1e7';ctx.lineWidth=2.3;ctx.lineJoin='round';ctx.stroke();
    if(!markers||markers.checked){
      ctx.fillStyle='#d7faff';
      for(let i=0;i<pts.length;i+=9){
        ctx.beginPath();ctx.arc(m.x(pts[i][0]),m.y(pts[i][1]),2.2,0,TAU);ctx.fill();
      }
    }
    ctx.restore();
    const error=Math.sqrt(pts.reduce((a,p,i)=>a+Math.pow(p[0]-initial[i][0],2)+Math.pow(p[1]-initial[i][1],2),0)/pts.length);
    if(timeOut)timeOut.textContent=t.toFixed(2)+' / 8.00';
    if(phaseOut)phaseOut.textContent=t<3.99?'STRETCHING':t<4.01?'FLOW REVERSAL':t<7.98?'RETURN FLOW':'CYCLE COMPLETE';
    if(returnOut)returnOut.textContent=error.toExponential(2);
    if(scrub)scrub.value=t.toFixed(2);
    if(originalNote)originalNote.textContent='Interface area / initial area: '+(contourArea(pts)/a0).toFixed(4)+' · point-displacement RMS: '+error.toExponential(2)+' (domain units)';
  }
  function pause(){playing=false;if(play){play.textContent='Play ▶';play.setAttribute('aria-pressed','false');}}
  function sync(){
    if(raf){cancelAnimationFrame(raf);raf=0;}
    if(playing&&visible&&!document.hidden&&!reduced.matches)raf=requestAnimationFrame(tick);
  }
  function tick(now){
    raf=0;
    if(!playing||!visible||document.hidden)return;
    const elapsed=last?Math.min(.055,(now-last)/1000):0;
    last=now;
    t=Math.min(T,t+elapsed*speedFactor);
    draw();
    if(t>=T){pause();return;}
    raf=requestAnimationFrame(tick);
  }
  if(play)play.addEventListener('click',()=>{
    if(playing){pause();sync();}
    else{if(t>=T)t=0;playing=true;last=0;play.textContent='Pause Ⅱ';play.setAttribute('aria-pressed','true');sync();}
  });
  if(reset)reset.addEventListener('click',()=>{pause();t=0;draw();sync();});
  if(step)step.addEventListener('click',()=>{pause();t=Math.min(T,t+.2);draw();sync();});
  if(scrub)scrub.addEventListener('input',()=>{pause();t=clamp(Number(scrub.value),0,T);draw();sync();});
  if(speed)speed.addEventListener('change',()=>{speedFactor=clamp(Number(speed.value),.25,4);});
  for(const cb of [vectors,markers])if(cb)cb.addEventListener('change',draw);
  if('IntersectionObserver' in window){
    new IntersectionObserver(es=>{visible=!!es[0]&&es[0].isIntersecting;last=0;sync();},{rootMargin:'100px 0px',threshold:0}).observe(canvas);
  }
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',()=>{last=0;sync();});
  if(reduced.addEventListener)reduced.addEventListener('change',()=>{if(reduced.matches)pause();sync();});
  if(reduced.matches)pause();
  resize();sync();
})();
'use strict';
/* REAL_FLUID WebAssembly front end. The numerical calculations are executed
 * by the author's C++ physics kernels, NOT by this script.
 * GPL-3.0: original source: Sagnik11October2004/REAL_FLUID.
 */
(() => {
  const q=id=>document.getElementById(id);
  const canvas=q('levelset-canvas');
  if(!canvas)return;
  const cx=canvas.getContext('2d',{alpha:false});
  if(!cx)return;
  const status=q('levelset-status'),run=q('levelset-run'),step=q('levelset-step'),
    reset=q('levelset-reset'),grid=q('levelset-grid'),vectors=q('levelset-vectors'),
    gravity=q('levelset-gravity'),sigma=q('levelset-sigma'),
    gravValue=q('levelset-gravity-value'),sigValue=q('levelset-sigma-value'),
    time=q('levelset-time'),area=q('levelset-area'),residual=q('levelset-divergence'),
    correction=q('levelset-volume-correct'),redistance=q('levelset-redistance'),
    solverSelect=q('levelset-pressure-solver'),interpSelect=q('levelset-interpolation'),
    resolution=q('levelset-resolution'),
    speed=q('levelset-speed'),description=q('levelset-description'),
    errorNote=q('native-fluid-build-error');
  const tabs=[...document.querySelectorAll('[data-levelset-case]')];
  const desc={
    dam:'Dam break: a water column initially occupies x < 0.4 and y < 0.6. Gravity drives the free-surface collapse.',
    drop:'Droplet impact: a liquid drop of radius 0.15, centered at (0.5, 0.8), falls toward a shallow liquid pool at y < 0.2.'
  };
  const modeNo={dam:0,drop:1};
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  let model=null,mode='dam',playing=!reduced.matches,visible=false,last=0,
      accumulator=0,raf=0,initialVolume=1,initialized=false,loading=false,coreReady=false;
  const setStatus=s=>{if(status)status.textContent=s;};
  function syncPlay(){
    if(run){
      run.textContent=playing?'Pause Ⅱ':'Play ▶';
      run.setAttribute('aria-pressed',String(playing));
      run.disabled=!model;
    }
  }
  function enableControls(ready){
    for(const el of [run,step,reset,gravity,sigma,correction,redistance,solverSelect,interpSelect,resolution,speed,...tabs])
      if(el)el.disabled=!ready;
  }
  enableControls(false);syncPlay();
  const prefix='./native-fluid/';
  async function loadModule(){
    if(loading||initialized)return;
    loading=true;enableControls(false);
    setStatus('LOADING ORIGINAL C++ / WASM SOLVER');
    try{
      if(!coreReady){
        await new Promise((resolve,reject)=>{
          const tag=document.createElement('script');
          tag.src=prefix+'fluid-core.js';
          tag.async=true;
          tag.onload=resolve;
          tag.onerror=()=>reject(new Error('Compiled WebAssembly loader is missing.'));
          document.head.appendChild(tag);
        });
        coreReady=true;
      }
      if(typeof createFluidModule!=='function')throw Error('Native solver factory was not exported.');
      const fresh=await createFluidModule({
        locateFile:f=>prefix+f,
        print:()=>{},printErr:()=>{}
      });
      const target=Number(resolution?.value||64);
      if(target!==64&&target!==128)throw Error('Unsupported grid size.');
      fresh._fluid_init(target);
      if(fresh._fluid_grid()!==target||!fresh.HEAPF32)
        throw Error('Native solver exports or Float32 heap not available.');
      // The original source allocates its arrays at initialization, so a grid
      // change creates a fresh WASM instance rather than resizing in place.
      model=fresh;
      if(mode!=='dam')model._fluid_choose_case(modeNo[mode]);
      if(correction&&!correction.checked)model._fluid_toggle_volume(0);
      if(redistance&&!redistance.checked)model._fluid_toggle_redistance(0);
      if(solverSelect)model._fluid_set_solver(Number(solverSelect.value));
      if(interpSelect)model._fluid_set_interpolation(Number(interpSelect.value));
      initialVolume=Math.abs(model._fluid_volume())||1;
      initialized=true;loading=false;
      enableControls(true);
      setPhysics();
      if(desc[mode]&&description)description.textContent=desc[mode];
      setStatus('FULL C++ ENGINE · '+target+'²');
      redraw();syncPlay();
      schedule();
    }catch(err){
      loading=false;
      initialized=false;model=null;playing=false;syncPlay();
      setStatus('FULL C++ ENGINE UNAVAILABLE');
      if(errorNote){
        errorNote.hidden=false;
        errorNote.textContent='The compiled REAL_FLUID WebAssembly engine is not available: '+String(err.message||err)+'. See the native C++ source and recording below.';
      }
    }
  }
  function setPhysics(){
    const g=Number(gravity?.value||9.8);
    const s=Number(sigma?.value||0.07);
    if(gravValue)gravValue.textContent=g.toFixed(1);
    if(sigValue)sigValue.textContent=s.toFixed(3);
    if(model)model._fluid_set_physics(g,s);
  }
  function setMode(value){
    if(!(value in modeNo))return;
    mode=value;
    tabs.forEach(b=>{
      const active=b.dataset.levelsetCase===mode;
      b.classList.toggle('active',active);
      b.setAttribute('aria-pressed',String(active));
    });
    if(description)description.textContent=desc[mode];
    if(!model)return;
    model._fluid_choose_case(modeNo[mode]);
    initialVolume=Math.abs(model._fluid_volume())||1;
    setPhysics();redraw();last=0;accumulator=0;
  }
  const liquidColor=[10,82,124],bgColor=[4,22,37];
  function drawLevelSet(){
    const n=model._fluid_grid();
    const ptr=model._fluid_phi_ptr()>>>2;
    const heap=model.HEAPF32;
    if(ptr+n*n>heap.length)throw Error('Native level-set buffer out of bounds');
    const phi=heap.subarray(ptr,ptr+n*n);
    const W=canvas.width,H=canvas.height;
    const side=Math.min(W-35,H-35),ox=(W-side)*.5,oy=(H-side)*.5;
    cx.fillStyle='#061522';cx.fillRect(0,0,W,H);
    const px=side/n;
    for(let j=0;j<n;j++)for(let i=0;i<n;i++){
      const f=phi[i+j*n];
      if(!Number.isFinite(f))throw Error('Non-finite native level-set value');
      const intensity=clamp(1.15-Math.abs(f)*14,0,1);
      if(f<0)cx.fillStyle='rgb('+Math.round(liquidColor[0]+40*intensity)+','+
          Math.round(liquidColor[1]+65*intensity)+','+Math.round(liquidColor[2]+69*intensity)+')';
      else cx.fillStyle='rgb('+bgColor[0]+','+bgColor[1]+','+bgColor[2]+')';
      cx.fillRect(ox+i*px,oy+(n-j-1)*px,px+.25,px+.25);
    }
    cx.beginPath();cx.strokeStyle='#a2f0fa';cx.lineWidth=Math.max(1.5,W/550);
    for(let j=0;j<n-1;j++)for(let i=0;i<n-1;i++){
      const p=i+j*n,a=phi[p],b=phi[p+1],c=phi[p+1+n],d=phi[p+n];
      const v=[[i+.5,j+.5],[i+1.5,j+.5],[i+1.5,j+1.5],[i+.5,j+1.5]];
      const vals=[a,b,c,d];const crossings=[];
      for(let k=0;k<4;k++){
        const z=(k+1)%4;
        if((vals[k]<0)!==(vals[z]<0)){
          const t=clamp(vals[k]/(vals[k]-vals[z]),0,1);
          crossings.push([v[k][0]+t*(v[z][0]-v[k][0]),v[k][1]+t*(v[z][1]-v[k][1])]);
        }
      }
      for(let k=0;k+1<crossings.length;k+=2){
        const x=crossings[k],y=crossings[k+1];
        cx.moveTo(ox+x[0]*px,oy+side-x[1]*px);
        cx.lineTo(ox+y[0]*px,oy+side-y[1]*px);
      }
    }
    cx.stroke();
    if(grid?.checked){
      cx.strokeStyle='rgba(156,217,236,.1)';cx.lineWidth=1;
      for(let a=0;a<=8;a++){
        const s=a*side/8;cx.beginPath();cx.moveTo(ox+s,oy);cx.lineTo(ox+s,oy+side);cx.stroke();
        cx.beginPath();cx.moveTo(ox,oy+s);cx.lineTo(ox+side,oy+s);cx.stroke();
      }
    }
    // Read the actual staggered native MAC field averaged to cell centers.
    if(vectors&&vectors.checked){
      const ui=model._fluid_u_ptr()>>>2,vi=model._fluid_v_ptr()>>>2;
      if(ui+n*n<=heap.length&&vi+n*n<=heap.length){
        const u=heap.subarray(ui,ui+n*n),v=heap.subarray(vi,vi+n*n);
        cx.strokeStyle='rgba(225,246,249,.62)';cx.fillStyle='rgba(225,246,249,.68)';
        cx.lineWidth=1.2;
        for(let j=3;j<n-3;j+=6)for(let i=3;i<n-3;i+=6){
          const k=i+n*j;if(phi[k]>=0)continue;
          const ux=u[k],vy=v[k],mag=Math.hypot(ux,vy);
          if(!Number.isFinite(mag)||mag<.02)continue;
          const size=Math.min(4.3,mag*2.5)*px;
          const x=ox+(i+.5)*px,y=oy+side-(j+.5)*px;
          const xx=x+ux/mag*size,yy=y-vy/mag*size;
          cx.beginPath();cx.moveTo(x,y);cx.lineTo(xx,yy);cx.stroke();
          const angle=Math.atan2(yy-y,xx-x);
          cx.beginPath();cx.moveTo(xx,yy);
          cx.lineTo(xx-4*Math.cos(angle-.5),yy-4*Math.sin(angle-.5));
          cx.lineTo(xx-4*Math.cos(angle+.5),yy-4*Math.sin(angle+.5));
          cx.closePath();cx.fill();
        }
      }
    }
    cx.strokeStyle='rgba(165,210,229,.57)';cx.lineWidth=1.2;
    cx.strokeRect(ox,oy,side,side);
  }
  function redraw(){
    if(!model)return;
    try{
      drawLevelSet();
      if(time)time.textContent=model._fluid_time().toFixed(2);
      if(area){
        const v=model._fluid_volume();
        area.textContent=Number.isFinite(v)?(Math.abs(v)/initialVolume*100).toFixed(1)+'%':'—';
      }
      if(residual){
        const v=model._fluid_pressure_residual();
        residual.textContent=Number.isFinite(v)?v.toExponential(2):'—';
      }
      setStatus('NATIVE C++ / WASM · '+model._fluid_time().toFixed(2)+' s · '+mode.toUpperCase());
    }catch(err){
      playing=false;syncPlay();
      setStatus('NATIVE SOLVER STOPPED');
      if(errorNote){errorNote.hidden=false;errorNote.textContent=String(err.message||err);}
    }
  }
  function resize(){
    const w=Math.max(310,Math.round(canvas.getBoundingClientRect().width||580));
    const dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);
    canvas.height=Math.round(w*.69*dpr);
    redraw();
  }
  function schedule(){
    if(raf){cancelAnimationFrame(raf);raf=0;}
    if(visible&&playing&&!document.hidden&&model)raf=requestAnimationFrame(frame);
  }
  function frame(now){
    raf=0;
    if(!playing||!visible||!model)return;
    let dt=last?Math.min((now-last)*.001,.06):0;last=now;
    accumulator+=dt*Number(speed?.value||1);
    let advanced=0;
    while(accumulator>=.01&&advanced<2){
      model._fluid_run_steps(1);
      accumulator-=.01;advanced++;
    }
    if(advanced)redraw();
    // The native solver has no artificial four-second simulation cutoff.
    schedule();
  }
  tabs.forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.levelsetCase)));
  if(run)run.addEventListener('click',()=>{playing=!playing;syncPlay();last=0;schedule();});
  if(step)step.addEventListener('click',()=>{
    if(!model)return;playing=false;syncPlay();model._fluid_run_steps(1);redraw();
  });
  if(reset)reset.addEventListener('click',()=>{
    if(!model)return;model._fluid_reset();initialVolume=Math.abs(model._fluid_volume())||1;
    setPhysics();redraw();last=0;accumulator=0;
  });
  for(const el of [gravity,sigma])if(el)el.addEventListener('input',setPhysics);
  if(correction)correction.addEventListener('change',()=>model?model._fluid_toggle_volume(correction.checked?1:0):null);
  if(redistance)redistance.addEventListener('change',()=>model?model._fluid_toggle_redistance(redistance.checked?1:0):null);
  if(solverSelect)solverSelect.addEventListener('change',()=>model?model._fluid_set_solver(Number(solverSelect.value)):null);
  if(interpSelect)interpSelect.addEventListener('change',()=>model?model._fluid_set_interpolation(Number(interpSelect.value)):null);
  if(resolution)resolution.addEventListener('change',()=>{
    if(loading)return;
    playing=false;syncPlay();model=null;initialized=false;last=0;accumulator=0;
    loadModule();
  });
  if(grid)grid.addEventListener('change',redraw);
  if(vectors)vectors.addEventListener('change',redraw);
  if('IntersectionObserver' in window)
    new IntersectionObserver(es=>{
      visible=!!es[0]&&es[0].isIntersecting;
      if(visible&&!initialized)loadModule();
      last=0;schedule();
    },{rootMargin:'180px 0px',threshold:0}).observe(canvas);
  else{visible=true;loadModule();}
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',()=>{last=0;schedule();});
  if(reduced.addEventListener)reduced.addEventListener('change',()=>{if(reduced.matches){playing=false;syncPlay();schedule();}});
  setMode(mode);resize();
})();
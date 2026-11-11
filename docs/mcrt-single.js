'use strict';
/*
 * One photon at a time, source-informed Monte Carlo Compton scattering demo.
 *
 * Scientific algorithms ported from the user's Nordita MCRT methodology:
 *  - optical-depth sampling tau=-ln(U), integrated through piecewise opacity;
 *  - interaction-weighted Maxwell–Jüttner / power-law electrons;
 *  - lab→electron-frame Lorentz boost, exact Klein–Nishina ERF rejection,
 *    Compton recoil and inverse Lorentz boost, escape from a spherical domain.
 *
 * Educational standalone JS, not the user's validated C/MPI transport solver.
 * Packets trace genuine 3D scattering flight histories but are projected into
 * an illustrative 2D image. Only H and S0 geometries.
 */
(() => {
  const canvas=document.getElementById('mcrt-scene');
  if(!canvas||!canvas.getContext)return;
  const c=canvas.getContext('2d',{alpha:false,desynchronized:true});
  if(!c)return;

  const TAU=2*Math.PI,ME=511,EPS=1e-12,clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const mul=(a,v)=>[a[0]*v,a[1]*v,a[2]*v];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2]];
  const norm=a=>Math.sqrt(dot(a,a));
  const unit=a=>mul(a,1/Math.max(norm(a),EPS));
  let seed=208110;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return Math.max(seed/4294967296,1e-12);};
  function iso(){
    const mu=2*random()-1,phi=TAU*random(),s=Math.sqrt(1-mu*mu);
    return [s*Math.cos(phi),s*Math.sin(phi),mu];
  }

  // Total Klein–Nishina cross section normalized by sigma_T.
  function sigmaKN(e){
    if(e<1e-4)return clamp(1-2*e+5.2*e*e,0,1);
    const l=Math.log1p(2*e),d=1+2*e;
    const t=(1+e)/Math.pow(e,3)*(2*e*(1+e)/d-l)+l/(2*e)-(1+3*e)/(d*d);
    return clamp(.75*t,0,1);
  }

  function lorentz(k,E,b){
    const b2=dot(b,b);
    if(b2<1e-12)return{k:k.slice(),E};
    const gamma=1/Math.sqrt(Math.max(1e-10,1-b2));
    const bdk=dot(b,k),D=Math.max(1e-9,gamma*(1-bdk));
    const coef=(gamma-1)*bdk/b2-gamma;
    return{k:unit(mul(add(k,mul(b,coef)),1/D)),E:E*D};
  }

  // Maxwell–Jüttner gamma density proportional gamma sqrt(gamma^2−1)
  // exp(−gamma/theta), sampled with a three-component Gamma-mixture envelope.
  function sampleMJ(theta){
    const a=2*theta,b=3*theta*theta,d=2*theta**3,total=a+b+d;
    for(let trial=0;trial<120;trial++){
      const U=random()*total,shape=U<a?1:U<a+b?2:3;
      let x=0;
      for(let i=0;i<shape;i++)x-=theta*Math.log(random());
      if(random()<Math.sqrt(x/(x+2)))return 1+x;
    }
    return 1+theta;
  }
  function powerlaw(){
    const lo=1.4,hi=6.0,p=2.8,U=random();
    return Math.pow(lo**(1-p)+U*(hi**(1-p)-lo**(1-p)),1/(1-p));
  }
  function electron(k,E,nonthermal){
    for(let n=0;n<160;n++){
      const gamma=nonthermal?powerlaw():sampleMJ(.196); // kTe≈100 keV
      const beta=Math.sqrt(Math.max(0,1-1/(gamma*gamma)));
      const v=mul(iso(),beta),dop=1-dot(v,k);
      const weight=clamp(dop*sigmaKN((E/ME)*gamma*dop)/2,0,1);
      if(random()<weight)return v;
    }
    return [0,0,0];
  }
  function compton(k,E,beta){
    const rest=lorentz(k,E,beta),eps=rest.E/ME;
    let mu=0,phi=0,ratio=1;
    for(let trial=0;trial<400;trial++){
      mu=2*random()-1;
      phi=TAU*random();
      ratio=1/(1+eps*(1-mu));
      const sinSq=1-mu*mu;
      const kn=ratio*ratio*(ratio+1/ratio-sinSq);
      if(random()<.5*kn)break;
    }
    const e1=unit(cross(rest.k,Math.abs(rest.k[2])<.9?[0,0,1]:[1,0,0]));
    const e2=cross(rest.k,e1),s=Math.sqrt(Math.max(0,1-mu*mu));
    const k2=unit(add(mul(rest.k,mu),add(mul(e1,s*Math.cos(phi)),mul(e2,s*Math.sin(phi)))));
    const out=lorentz(k2,rest.E*ratio,mul(beta,-1));
    return{k:out.k,E:clamp(out.E,.02,5000)};
  }

  // A fixed cloud catalogue illustrates static plasmoids in S0.
  const blobs=Array.from({length:22},(_,i)=>{
    const a=i*2.399963,r=.14+.60*Math.sqrt((i+.4)/22);
    return {p:[r*Math.cos(a),r*Math.sin(a),.14*Math.sin(2.3*i)],r:.105+(i%5)*.012};
  });
  function inBlob(r){return blobs.some(b=>norm(sub(b.p,r))<b.r);}

  let mode='H',W=900,H=550,DPR=1,playing=true,visible=true,raf=0,last=0,paint=0,elapsed=0,velocity=1;
  let total=0,escaped=0,scattered=0,counts=new Uint32Array(28),packet=null,flight=0,flightTime=0;
  let dwell=.9,completed=false,lastEvent='Ready to inject seed photon';
  const counter=document.getElementById('mcrt-count'),summary=document.getElementById('mcrt-spectrum-summary');
  const buttons=[...document.querySelectorAll('.mcrt-mode')],play=document.getElementById('mcrt-play');
  const resetButton=document.getElementById('mcrt-reset'),next=document.getElementById('mcrt-next');
  const speedInput=document.getElementById('mcrt-speed');
  const reduced=window.matchMedia('(prefers-reduced-motion:reduce)');
  playing=!reduced.matches;

  function boundary(r,k){
    const B=dot(r,k),d=B*B+1-dot(r,r);
    return Math.max(0,-B+Math.sqrt(Math.max(0,d)));
  }
  function flightOpacity(r,E){
    const inside=mode==='S0'&&inBlob(r);
    const n=mode==='H'?3.1:inside?7.2:1.10;
    return {alpha:n*sigmaKN(E/ME),nonthermal:mode==='S0'&&!inside};
  }
  function computePhoton(){
    let r=mul(iso(),.018),k=iso(),E=.8+.45*random();
    const emittedEnergy=E,segments=[],maxScatter=55;
    for(let scatterCount=0;scatterCount<=maxScatter;scatterCount++){
      const start=r.slice(),startE=E,distMax=boundary(r,k);
      let tauLeft=-Math.log(random()),dTotal=0,hit=false,nonthermal=false;
      const steps=Math.ceil(distMax/.018)+1;
      for(let j=0;j<steps;j++){
        const bd=boundary(r,k),ds=Math.min(.018,bd);
        if(ds<EPS)break;
        const op=flightOpacity(r,E),tau=op.alpha*ds;
        if(tauLeft<=tau&&op.alpha>0){
          const actual=tauLeft/op.alpha;
          r=add(r,mul(k,actual));dTotal+=actual;
          hit=true;nonthermal=op.nonthermal;break;
        }
        r=add(r,mul(k,ds));dTotal+=ds;tauLeft-=tau;
        if(bd<=ds+1e-7)break;
      }
      // Duration is perceptual, never a claim about photon physical flight time.
      const duration=clamp(.55+dTotal*1.65,.8,2.2);
      const entry={from:start,to:r.slice(),before:startE,after:E,scatter:hit,duration,
                   scatterIndex:scatterCount+1};
      if(hit&&scatterCount<maxScatter){
        const b=electron(k,E,nonthermal),out=compton(k,E,b);
        E=out.E;k=out.k;entry.after=E;
        segments.push(entry);
      }else{
        entry.scatter=false;entry.escaped=(norm(r)>=.99999);
        segments.push(entry);break;
      }
    }
    const out=segments[segments.length-1];
    return {id:++total,segments,emittedEnergy,finalEnergy:out?.after||E,
      escaped:!!out?.escaped,step:0};
  }
  const bin=E=>clamp(Math.floor(28*Math.log(Math.max(.15,E)/.15)/Math.log(400/.15)),0,27);
  function reset(){
    total=0;escaped=0;scattered=0;counts=new Uint32Array(28);
    packet=null;flight=0;flightTime=0;dwell=.05;completed=false;
    lastEvent='Ready to inject seed photon';draw();
  }
  function inject(){
    packet=computePhoton();flight=0;flightTime=0;completed=false;
    lastEvent='Packet '+packet.id+' injected · E₀='+packet.emittedEnergy.toFixed(2)+' keV';
  }
  function finish(){
    if(completed||!packet)return;
    completed=true;
    if(packet.escaped){
      escaped++;counts[bin(packet.finalEnergy)]++;
      lastEvent='Packet '+packet.id+' escaped · E='+packet.finalEnergy.toFixed(2)+' keV';
    }else lastEvent='Packet '+packet.id+' reached scattering limit (not counted)';
    dwell=1.05;
  }
  function advance(dt){
    if(!Number.isFinite(dt)||dt<=0)return;
    let left=dt*velocity;
    let loops=0;
    while(left>0&&loops++<30){
      if(!packet||completed){
        const time=Math.min(left,dwell);
        dwell-=time;left-=time;
        if(dwell<=1e-8)inject();
        else break;
      }
      if(!packet||completed)continue;
      const segment=packet.segments[flight];
      if(!segment){finish();continue;}
      const step=Math.min(left,Math.max(0,segment.duration-flightTime));
      flightTime+=step;left-=step;
      if(flightTime>=segment.duration-1e-8){
        if(segment.scatter){
          scattered++;
          lastEvent='Scatter '+segment.scatterIndex+' · '+segment.before.toFixed(2)+' → '+segment.after.toFixed(2)+' keV';
        }
        flight++;flightTime=0;
        if(flight>=packet.segments.length)finish();
      }else break;
    }
  }
  function drawCircle(x,y,r,col){c.beginPath();c.arc(x,y,Math.max(.1,r),0,TAU);c.fillStyle=col;c.fill();}
  function line(a,b,col,width=1){
    c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.strokeStyle=col;c.lineWidth=width;c.stroke();
  }
  const xy=v=>[W*(.397+.29*v[0]),H*(.52-.37*v[1])]; // orthographic 3D projection
  function color(E,a=1){
    const f=clamp(Math.log(E/.8)/Math.log(20/.8),0,1);
    return 'rgba('+Math.round(117+138*f)+','+Math.round(220-63*f)+','+Math.round(255-148*f)+','+a+')';
  }
  function glow(x,y,r,rgb,a){
    const g=c.createRadialGradient(x,y,0,x,y,r);
    g.addColorStop(0,'rgba('+rgb+','+a+')');g.addColorStop(1,'rgba('+rgb+',0)');
    c.fillStyle=g;c.fillRect(x-r,y-r,2*r,2*r);
  }
  function render(){
    const g=c.createLinearGradient(0,0,W,H);g.addColorStop(0,'#030917');g.addColorStop(.55,'#092039');g.addColorStop(1,'#030814');
    c.fillStyle=g;c.fillRect(0,0,W,H);
    const center=xy([0,0,0]),R=Math.min(W*.285,H*.37);
    glow(center[0],center[1],R*1.65,'57,109,177',.33);
    c.beginPath();c.arc(center[0],center[1],R,0,TAU);
    c.fillStyle='rgba(30,99,146,.12)';c.fill();c.strokeStyle='rgba(128,211,245,.40)';c.lineWidth=1.5;c.stroke();
    c.beginPath();c.ellipse(center[0],center[1],R,R*.27,0,0,TAU);
    c.strokeStyle='rgba(160,210,231,.13)';c.lineWidth=1;c.stroke();
    if(mode==='S0'){
      for(const b of blobs){
        const q=xy(b.p),r=b.r*R;
        glow(q[0],q[1],r*1.8,'62,149,209',.19);
        c.beginPath();c.arc(q[0],q[1],r,0,TAU);
        c.strokeStyle='rgba(127,224,245,.43)';c.lineWidth=1;c.stroke();
      }
    }else{
      for(let i=0;i<85;i++){
        const a=i*2.399963,rr=R*Math.sqrt((i+.5)/85);
        drawCircle(center[0]+rr*Math.cos(a),center[1]+rr*Math.sin(a),.55,'rgba(119,204,242,.28)');
      }
    }
    glow(center[0],center[1],20,'103,223,255',.27);
    drawCircle(center[0],center[1],3,'#e1fbff');

    if(packet){
      const n=Math.min(flight,packet.segments.length);
      for(let i=0;i<n;i++){
        const v=packet.segments[i];line(xy(v.from),xy(v.to),color(v.before,.60),2);
        if(v.scatter){
          const pt=xy(v.to);glow(pt[0],pt[1],15,'255,195,112',.31);
          drawCircle(pt[0],pt[1],2.4,'rgba(255,197,128,.9)');
        }
      }
      const segment=packet.segments[Math.min(flight,packet.segments.length-1)];
      if(segment){
        const prog=completed?1:clamp(flightTime/segment.duration,0,1);
        const at=add(segment.from,mul(sub(segment.to,segment.from),prog));
        const pp=xy(at);
        if(!completed){
          line(xy(segment.from),pp,color(segment.before,.90),2.2);
          glow(pp[0],pp[1],20,segment.before>2?'255,173,98':'95,210,255',.49);
          drawCircle(pp[0],pp[1],4.0,color(segment.before,1));
        }else if(packet.escaped){
          glow(pp[0],pp[1],25,'160,240,199',.43);
          drawCircle(pp[0],pp[1],3.4,'rgba(148,255,223,.95)');
        }
      }
    }
    const bx=W*.745,by=H*.20,bw=W*.222,bh=H*.61;
    c.fillStyle='rgba(3,17,32,.88)';c.fillRect(bx-9,by-30,bw+17,bh+63);
    c.strokeStyle='rgba(137,207,239,.25)';c.strokeRect(bx-9,by-30,bw+17,bh+63);
    c.textAlign='center';c.font='700 '+Math.max(10,W*.014)+'px system-ui';c.fillStyle='#cdeefa';
    c.fillText('ESCAPED PHOTON SPECTRUM',bx+bw/2,by-12);
    const max=Math.max(1,...counts),baseY=by+bh-39,upper=bh-62;
    for(let i=0;i<counts.length;i++){
      const hh=counts[i]/max*upper;
      c.fillStyle=color(.15*Math.exp((i+.5)/28*Math.log(400/.15)),.88);
      c.fillRect(bx+i*bw/28,baseY-hh,Math.max(1,bw/28-1.1),hh);
    }
    // Cumulative density estimate guides the eye when only a few packets escaped.
    const smoothed=Array.from({length:28},(_,j)=>{
      let y=0;for(let i=0;i<28;i++)y+=counts[i]*Math.exp(-.5*((j-i)/1.3)**2);
      return y;
    });
    const scaleY=Math.max(1,...smoothed);
    c.beginPath();for(let j=0;j<28;j++){
      const x=bx+(j+.5)*bw/28,y=baseY-smoothed[j]/scaleY*upper;
      if(j===0)c.moveTo(x,y);else c.lineTo(x,y);
    }
    c.lineWidth=1.6;c.strokeStyle='rgba(218,244,255,.79)';c.stroke();
    const ticks=[[.2,'0.2'],[1,'1'],[10,'10'],[100,'100']];
    c.font=Math.max(9,W*.011)+'px system-ui';c.fillStyle='#c3d4e8';
    for(const [E,label] of ticks){
      const x=bx+bw*Math.log(E/.15)/Math.log(400/.15);
      c.fillText(label,x,baseY+18);
    }
    c.fillText('Photon energy (keV) · log scale',bx+bw/2,baseY+40);

    if(counter)counter.textContent='Photon #'+(packet?.id||(total+1))+
      ' · '+escaped+' escaped · '+scattered+' scatters';
    if(summary)summary.textContent=lastEvent+' · '+mode+' corona';
  }
  function draw(){render();}
  let paintedAt=0;
  function frame(now){
    raf=0;if(!playing||document.hidden||!visible)return;
    const dt=last?clamp((now-last)/1000,0,.075):0;last=now;
    if(dt>0)advance(dt);
    if(now-paintedAt>40){render();paintedAt=now;}
    raf=requestAnimationFrame(frame);
  }
  function sync(){
    if(raf)cancelAnimationFrame(raf);raf=0;last=0;
    if(play){play.textContent=playing?'Pause Ⅱ':'Play ▶';play.setAttribute('aria-label',playing?'Pause photon trace':'Resume photon trace');}
    render();
    if(playing&&!document.hidden&&visible)raf=requestAnimationFrame(frame);
  }
  function size(){
    const rect=canvas.getBoundingClientRect();W=Math.max(275,rect.width);H=Math.max(215,rect.height);DPR=Math.min(window.devicePixelRatio||1,1.5);
    canvas.width=Math.round(W*DPR);canvas.height=Math.round(H*DPR);
    c.setTransform(DPR,0,0,DPR,0,0);render();
  }
  if(play)play.addEventListener('click',()=>{playing=!playing;sync();});
  if(resetButton)resetButton.addEventListener('click',()=>{reset();sync();});
  if(next)next.addEventListener('click',()=>{
    if(packet&&!completed){
      for(let i=flight;i<packet.segments.length;i++)if(packet.segments[i].scatter)scattered++;
      finish();
    }
    dwell=0;inject();sync();
  });
  if(speedInput)speedInput.addEventListener('change',()=>{velocity=clamp(Number(speedInput.value)||1,1,4);});
  buttons.forEach(button=>button.addEventListener('click',()=>{
    mode=button.dataset.mcrtMode==='S0'?'S0':'H';
    buttons.forEach(b=>{const selected=b===button;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));});
    reset();sync();
  }));
  if('IntersectionObserver' in window)new IntersectionObserver(es=>{
    visible=!!es[0]&&es[0].isIntersecting;sync();
  },{threshold:.01}).observe(canvas);
  window.addEventListener('resize',size,{passive:true});
  document.addEventListener('visibilitychange',sync);
  if(reduced.addEventListener)reduced.addEventListener('change',()=>{if(reduced.matches)playing=false;sync();});
  size();reset();sync();
})();

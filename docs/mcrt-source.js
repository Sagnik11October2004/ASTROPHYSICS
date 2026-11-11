'use strict';
/*
 * Nordita MCRT — public educational browser port.
 *
 * Based on methods inspected in the author's PRIVATE nordita-mcrt C source:
 *   src/transport.c:  sampled tau=-ln(U), flight lengths, boundary escape;
 *   src/physics.c:    Lorentz boost to and from the electron rest frame,
 *                     exact Klein–Nishina angular rejection kernel,
 *                     electron distribution and scattering component selection;
 *   blob/domain tools: structured and moving scattering regions.
 *
 * This is ORIGINAL JavaScript using the same published physical algorithms,
 * NOT a verbatim release of the private research code, not a full production
 * run and not a quantitative polarized spectrum. Polarization is not propagated.
 *
 * Electron distributions:
 * H:  Maxwell–Jüttner at 100 keV in a homogeneous spherical corona.
 * S0: Maxwell–Jüttner blobs + nonthermal power-law interblob electrons.
 * Sv: cooler moving plasmoids with an isotropic thermal spread and bulk motion.
 * All geometries are educational stand-ins, not fitted copies of paper cases.
 */
(() => {
 const canvas=document.getElementById('mcrt-scene');
 if(!canvas||!canvas.getContext)return;
 const c=canvas.getContext('2d',{alpha:false,desynchronized:true});
 if(!c)return;
 const TAU=2*Math.PI,me=511.0,EPS=1e-12;
 const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
 const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
 const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
 const scale=(a,k)=>[a[0]*k,a[1]*k,a[2]*k];
 const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const norm=a=>Math.sqrt(dot(a,a));
 const unit=a=>scale(a,1/Math.max(norm(a),EPS));
 let seed=208110;
 function rng(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return Math.max(1e-12,seed/4294967296);}
 function iso(){const m=2*rng()-1,phi=TAU*rng(),s=Math.sqrt(1-m*m);return [s*Math.cos(phi),s*Math.sin(phi),m];}
 function sigmaKN(e){
   if(e<1e-4)return Math.max(0,1-2*e+5.2*e*e);
   const l=Math.log1p(2*e),d=1+2*e;
   const t=(1+e)/(e**3)*(2*e*(1+e)/d-l)+l/(2*e)-(1+3*e)/(d*d);
   return clamp(.75*t,0,1);
 }
 function lorentzPhoton(k,E,beta){
   const b2=dot(beta,beta);
   if(b2<1e-12)return{k:k.slice(),E};
   const gam=1/Math.sqrt(Math.max(1e-7,1-b2)),d=dot(beta,k),dop=gam*(1-d);
   const coeff=(gam-1)*d/b2-gam;
   return{k:unit(scale(add(k,scale(beta,coeff)),1/Math.max(dop,1e-10))),E:E*dop};
 }
 function sampleGammaMJ(theta){
   // gamma^2 beta exp(-gamma/theta) in gamma, with x=gamma-1.
   // Propose x with mixture of Gamma(k=1,2,3) distributions:
   // (x+1)(x+2)exp(-x/theta), accept sqrt(x/(x+2)).
   const a=2*theta,b=3*theta*theta,d=2*theta*theta*theta,tot=a+b+d;
   for(let trials=0;trials<100;trials++){
     const u=rng()*tot,k=u<a?1:u<a+b?2:3;
     let x=0;for(let j=0;j<k;j++)x-=theta*Math.log(rng());
     if(rng()<Math.sqrt(x/(x+2)))return 1+x;
   }
   return 1+Math.max(1e-6,-theta*Math.log(rng()));
 }
 function samplePLGamma(){
   const lo=1.4,hi=6.0,q=2.8,u=rng();
   return Math.pow(Math.pow(lo,1-q)+u*(Math.pow(hi,1-q)-Math.pow(lo,1-q)),1/(1-q));
 }
 function weightedElectron(k,E,region,velocity){
   // Interaction-weighted proposal: (1-beta.k) * sigma_KN(E') .
   const theta=region==='cool'?.018:region==='blob'?.196:region==='nonthermal'?.24:.196;
   for(let i=0;i<90;i++){
     const gamma=region==='nonthermal'?samplePLGamma():sampleGammaMJ(theta);
     const beta=Math.sqrt(Math.max(0,1-1/(gamma*gamma)));
     const internal=scale(iso(),beta);
     const b=velocity?add(scale(internal,1-.22*norm(velocity)),velocity):internal;
     const limit=norm(b);
     const electron=limit>.94?scale(b,.94/limit):b;
     const vdot=dot(electron,k),g=1/Math.sqrt(1-dot(electron,electron));
     const eps=E/me*g*(1-vdot);
     const weight=Math.max(0,(1-vdot)*sigmaKN(eps)/2);
     if(rng()<weight)return electron;
   }
   return velocity||[0,0,0];
 }
 function scatter(k,E,beta){
   const pre=lorentzPhoton(k,E,beta),eps=pre.E/me;
   let mu=0,phi=0,eta=1;
   for(let attempt=0;attempt<300;attempt++){
     mu=2*rng()-1;phi=TAU*rng();
     eta=1/(1+eps*(1-mu));
     const kn=eta*eta*(eta+1/eta-(1-mu*mu));
     if(rng()<.5*kn)break;
   }
   const a=unit(cross(pre.k,Math.abs(pre.k[2])<.9?[0,0,1]:[1,0,0]));
   const b=cross(pre.k,a),s=Math.sqrt(Math.max(0,1-mu*mu));
   const direction=unit(add(scale(pre.k,mu),add(scale(a,s*Math.cos(phi)),scale(b,s*Math.sin(phi)))));
   const restOut=pre.E*eta;
   const lab=lorentzPhoton(direction,restOut,scale(beta,-1));
   return{k:lab.k,E:clamp(lab.E,.02,1e4)};
 }
 const blobs=Array.from({length:20},(_,i)=>{
   const a=i*2.399963,r=.15+.6*Math.sqrt((i+.5)/20);
   return{c:[r*Math.cos(a),r*Math.sin(a),.18*Math.sin(i*1.32)],r:.11+(i%4)*.019,
          spin:i*1.3, phase:i*.27};
 });
 function blobAt(pos,t){
   let closest=null,best=1e10;
   for(const b of blobs){
     const shift=mode==='Sv'?[Math.cos(t*.5+b.phase)*.044,Math.sin(t*.42+b.phase)*.044,0]:[0,0,0];
     const center=add(b.c,shift),d=norm(sub(pos,center));
     if(d<b.r && d<best){best=d;closest={b,center};}
   }
   return closest;
 }
 let mode='H',playing=true,visible=true,raf=0,last=0,paint=0,t=0;
 let W=900,H=550,DPR=1,packets=[],history=[],escapes=0,scatters=0,launched=0,meanEnergy=0,counts=new Uint32Array(26),maxScatter=70;
 const buttons=[...document.querySelectorAll('.mcrt-mode')];
 const count=document.getElementById('mcrt-count'),summary=document.getElementById('mcrt-spectrum-summary');
 const play=document.getElementById('mcrt-play'),resetButton=document.getElementById('mcrt-reset');
 const reduce=window.matchMedia('(prefers-reduced-motion:reduce)');
 playing=!reduce.matches;
 function spawn(){
   // Isotropic seed photons at the center of a spherical corona.
   const ph={r:scale(iso(),.022),k:iso(),E:.55+.8*rng(),tau:-Math.log(rng()),
     sc:0,track:[],age:0};
   launched++;return ph;
 }
 function reset(){
   packets=[];history=[];escapes=0;scatters=0;launched=0;meanEnergy=0;counts=new Uint32Array(26);t=0;
   for(let i=0;i<110;i++)packets.push(spawn());
   draw();
 }
 function opacity(ph){
   // Approximate energy-dependent extinction as in the C transport workflow.
   // The production solver integrates opacities and separate local components.
   const inside=blobAt(ph.r,t);
   const density=mode==='H'?1:inside?3.6:.16;
   return{inside,density,alpha:1.65*density*sigmaKN(ph.E/me),
      region:mode==='H'?'blob':mode==='Sv'&&inside?'cool':inside?'blob':'nonthermal'};
 }
 function boundaryDistance(r,k){
   const z=dot(r,k),D=z*z+1-dot(r,r);
   return Math.max(0,-z+Math.sqrt(Math.max(0,D)));
 }
 function outputBin(e){
   return clamp(Math.floor(26*Math.log(Math.max(.08,e)/.08)/Math.log(400/.08)),0,25);
 }
 function advance(dt){
   if(!Number.isFinite(dt)||dt<=0)return;
   t+=dt;
   const nsteps=clamp(Math.ceil(dt*190),1,12),ds=.033;
   for(let step=0;step<nsteps;step++){
     for(let i=packets.length-1;i>=0;i--){
       const ph=packets[i];
       if(ph.age>12||ph.sc>=maxScatter){packets.splice(i,1);continue;}
       ph.age+=ds;
       const op=opacity(ph),dBorder=boundaryDistance(ph.r,ph.k);
       const flight=Math.min(ds,dBorder);
       const tau=op.alpha*flight;
       const old=ph.r.slice();
       if(ph.tau<=tau&&op.alpha>0){
         const actual=ph.tau/op.alpha;ph.r=add(old,scale(ph.k,actual));
         ph.track.push({from:old,to:ph.r.slice(),E:ph.E});
         const local=op.inside,bulk=mode==='Sv'&&local?
           scale(unit([-local.center[1],local.center[0],.09]),.33):null;
         const electron=weightedElectron(ph.k,ph.E,op.region,bulk);
         const newPacket=scatter(ph.k,ph.E,electron);
         ph.k=newPacket.k;ph.E=newPacket.E;ph.tau=-Math.log(rng());ph.sc++;scatters++;
         history.push({r:ph.r.slice(),E:ph.E,ttl:.38});
         if(ph.track.length>13)ph.track.shift();
       }else{
         ph.r=add(old,scale(ph.k,flight));ph.tau-=tau;
         ph.track.push({from:old,to:ph.r.slice(),E:ph.E});
         if(ph.track.length>13)ph.track.shift();
         if(dBorder<=ds+1e-8){
           escapes++;meanEnergy+=(ph.E-meanEnergy)/escapes;counts[outputBin(ph.E)]++;
           packets.splice(i,1);
         }
       }
     }
     for(let n=0;n<3;n++)if(packets.length<175 && launched<40000)packets.push(spawn());
   }
   history=history.filter(v=>(v.ttl-=dt)>0);
 }
 function particleColor(E,a){
   const f=clamp(Math.log(E/.6)/Math.log(30/.6),0,1);
   return 'rgba('+Math.round(108+147*f)+','+Math.round(202-58*f)+','+Math.round(255-150*f)+','+a+')';
 }
 function xy(r){
   const sx=W*.11+W*.65*(.5+r[0]*.41),sy=H*.12+H*.79*(.5-r[1]*.41);
   return[sx,sy];
 }
 function line(ax,ay,bx,by,color,lw){
   c.beginPath();c.moveTo(ax,ay);c.lineTo(bx,by);c.strokeStyle=color;c.lineWidth=lw;c.stroke();
 }
 function dot2(x,y,r,color){c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fillStyle=color;c.fill();}
 function glow(x,y,r,rgba){
   const g=c.createRadialGradient(x,y,0,x,y,r);
   g.addColorStop(0,rgba);g.addColorStop(1,'rgba(2,12,23,0)');
   c.fillStyle=g;c.fillRect(x-r,y-r,2*r,2*r);
 }
 function render(){
   const bg=c.createLinearGradient(0,0,W,H);bg.addColorStop(0,'#020819');bg.addColorStop(.5,'#0b2744');bg.addColorStop(1,'#020a16');
   c.fillStyle=bg;c.fillRect(0,0,W,H);
   const cx=W*.435,cy=H*.515,R=Math.min(W*.33,H*.385);
   glow(cx,cy,R*1.7,'rgba(49,120,187,.24)');
   c.beginPath();c.arc(cx,cy,R,0,TAU);
   c.fillStyle='rgba(25,81,124,.12)';c.fill();
   c.strokeStyle='rgba(125,219,253,.34)';c.lineWidth=1.4;c.stroke();
   for(let k=0;k<8;k++){
     const phi=k*TAU/8;
     c.beginPath();c.ellipse(cx,cy,R*Math.cos(phi*.09),R*.28,phi*.1,0,TAU);
     c.strokeStyle='rgba(129,185,224,.09)';c.lineWidth=.9;c.stroke();
   }
   if(mode==='H'){
     for(let i=0;i<155;i++){
       const v=(i*.61803398)%1,a=i*2.399963,rr=R*Math.sqrt(v)*.94;
       dot2(cx+Math.cos(a)*rr,cy+Math.sin(a)*rr,.5,'rgba(104,191,238,.18)');
     }
   }else{
     for(let i=0;i<blobs.length;i++){
       const b=blobs[i],shift=mode==='Sv'?[Math.cos(t*.5+b.phase)*.044,Math.sin(t*.42+b.phase)*.044,0]:[0,0,0];
       const pt=xy(add(b.c,shift)),r=b.r*R;
       glow(pt[0],pt[1],r*1.6,mode==='Sv'?'rgba(59,201,226,.2)':'rgba(100,168,245,.23)');
       c.beginPath();c.arc(pt[0],pt[1],r,0,TAU);
       c.strokeStyle='rgba(144,222,248,.39)';c.lineWidth=1.2;c.stroke();
       if(mode==='S0' && i%5===0)glow(pt[0]+r*.6,pt[1]-r*.3,r*.65,'rgba(245,126,91,.16)');
     }
   }
   const src=xy([0,0,0]);glow(src[0],src[1],24,'rgba(124,231,255,.50)');
   dot2(src[0],src[1],3,'#d1faff');
   for(const ph of packets){
     for(let i=0;i<ph.track.length;i++){
       const step=ph.track[i],a=xy(step.from),b=xy(step.to);
       line(a[0],a[1],b[0],b[1],particleColor(step.E,.17+.68*(i+1)/ph.track.length),1.45);
     }
     const a=xy(ph.r);glow(a[0],a[1],8,particleColor(ph.E,.18));
     dot2(a[0],a[1],2.25,particleColor(ph.E,.96));
   }
   for(const hit of history){
     const p=xy(hit.r),r=(.42-hit.ttl)*36;
     c.beginPath();c.arc(p[0],p[1],Math.max(1,r),0,TAU);
     c.strokeStyle=particleColor(hit.E,hit.ttl*.75);c.lineWidth=1.2;c.stroke();
   }
   const bx=W*.795,by=H*.22,bw=W*.17,bh=H*.50;
   c.fillStyle='rgba(2,15,30,.90)';c.fillRect(bx-7,by-39,bw+16,bh+76);
   c.strokeStyle='rgba(104,194,239,.23)';c.strokeRect(bx-7,by-39,bw+16,bh+76);
   c.textAlign='center';c.font='700 '+Math.max(10,W*.013)+'px system-ui';c.fillStyle='#c6e6ec';
   c.fillText('ESCAPED ENERGY',bx+bw*.5,by-17);
   const max=Math.max(1,...counts);
   for(let i=0;i<counts.length;i++){
     const height=counts[i]/max*(bh-40);
     c.fillStyle=particleColor(.08*Math.exp((i+.5)/26*Math.log(400/.08)),.84);
     c.fillRect(bx+i*bw/26,by+bh-30-height,Math.max(2,bw/26-.8),height);
   }
   c.fillStyle='#b2cad8';c.font=Math.max(9,W*.010)+'px system-ui';
   c.fillText('LOW E',bx+bw*.22,by+bh-7);c.fillText('HIGH E',bx+bw*.80,by+bh-7);
   if(count)count.textContent=escapes+' escaped · '+scatters+' scatterings · '+launched+' injected';
   if(summary)summary.textContent=mode==='H'?'H: thermal 100 keV, homogeneous sphere':
     mode==='S0'?'S0: thermal blobs / nonthermal interblob regions':
     'Sv: cooler plasmoids with bulk motion (β≈0.33)';
 }
 function draw(){render();}
 function frame(now){
   raf=0;if(!playing||!visible||document.hidden)return;
   const dt=last?clamp((now-last)/1000,0,.05):0;last=now;
   if(dt)advance(dt);
   if(now-paint>40){render();paint=now;}
   raf=requestAnimationFrame(frame);
 }
 function sync(){
   if(raf)cancelAnimationFrame(raf);raf=0;last=0;paint=0;
   if(play){play.textContent=playing?'Pause Ⅱ':'Play ▶';play.setAttribute('aria-label',playing?'Pause source-based Monte Carlo':'Play source-based Monte Carlo');}
   render();if(playing&&!document.hidden&&visible)raf=requestAnimationFrame(frame);
 }
 function size(){
   const r=canvas.getBoundingClientRect();W=Math.max(280,r.width);H=Math.max(240,r.height);
   DPR=Math.min(window.devicePixelRatio||1,1.5);
   canvas.width=Math.round(W*DPR);canvas.height=Math.round(H*DPR);
   c.setTransform(DPR,0,0,DPR,0,0);render();
 }
 if(play)play.addEventListener('click',()=>{playing=!playing;sync();});
 const resetButtonLocal=resetButton;
 if(resetButtonLocal)resetButtonLocal.addEventListener('click',()=>{reset();sync();});
 buttons.forEach(b=>b.addEventListener('click',()=>{
   mode=b.dataset.mcrtMode;
   buttons.forEach(other=>{
     const active=other===b;other.classList.toggle('active',active);
     other.setAttribute('aria-pressed',String(active));
   });
   reset();sync();
 }));
 if('IntersectionObserver' in window)new IntersectionObserver(es=>{visible=!!es[0]&&es[0].isIntersecting;sync();},{threshold:.02}).observe(canvas);
 window.addEventListener('resize',size,{passive:true});
 document.addEventListener('visibilitychange',sync);
 if(reduce.addEventListener)reduce.addEventListener('change',()=>{if(reduce.matches)playing=false;sync();});
 size();reset();sync();
})();

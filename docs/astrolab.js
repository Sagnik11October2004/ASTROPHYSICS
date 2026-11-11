'use strict';
/* Research animation suite: canvas educational models only.
   AGN: gas supply / outflow / heating / renewed fuel (not GRMHD).
   Pop III.1: the uploaded four-stage comparison video is the storyboard;
     annihilation-powered supermassive growth remains hypothetical.
   MCRT: real stochastic packet scattering with approximate energy gain,
     not a calibrated polarization/radiative-transfer solution. */
(() => {
 const TAU=2*Math.PI,clamp=(v,a,b)=>Math.min(b,Math.max(a,v)),mix=(a,b,t)=>a+(b-a)*t,
 smooth=v=>{v=clamp(v,0,1);return v*v*(3-2*v);},reduce=window.matchMedia('(prefers-reduced-motion: reduce)');
 let rngSeed=718877;
 function rnd(){rngSeed=(Math.imul(rngSeed,1664525)+1013904223)>>>0;return rngSeed/4294967296;}
 function context(id){
   const canvas=document.getElementById(id);
   if(!canvas||!canvas.getContext)return null;
   const ctx=canvas.getContext('2d',{alpha:false,desynchronized:true});
   if(!ctx)return null;
   return {canvas,ctx,w:800,h:500,visible:true,scale:1};
 }
 function resize(scene){
   const r=scene.canvas.getBoundingClientRect();
   scene.w=Math.max(260,r.width);scene.h=Math.max(220,r.height);
   const dpr=Math.min(window.devicePixelRatio||1,1.5);
   scene.canvas.width=Math.round(scene.w*dpr);scene.canvas.height=Math.round(scene.h*dpr);
   scene.ctx.setTransform(dpr,0,0,dpr,0,0);
 }
 function circle(c,x,y,r,color){c.beginPath();c.arc(x,y,Math.max(.2,r),0,TAU);c.fillStyle=color;c.fill();}
 function glow(c,x,y,r,rgb,alpha){
   const g=c.createRadialGradient(x,y,0,x,y,r);
   g.addColorStop(0,'rgba('+rgb+','+alpha+')');g.addColorStop(.35,'rgba('+rgb+','+(alpha*.38)+')');g.addColorStop(1,'rgba('+rgb+',0)');
   c.fillStyle=g;c.fillRect(x-r,y-r,r*2,r*2);
 }
 function stroke(c,x1,y1,x2,y2,color,width=1){
   c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.lineWidth=width;c.strokeStyle=color;c.stroke();
 }
 function base(s){
   const c=s.ctx,g=c.createLinearGradient(0,0,s.w,s.h);g.addColorStop(0,'#020817');g.addColorStop(.5,'#0a1e31');g.addColorStop(1,'#020915');
   c.fillStyle=g;c.fillRect(0,0,s.w,s.h);
   for(let k=0;k<98;k++){
     let x=((k*157.73)%997)/997*s.w,y=((k*369.15)%991)/991*s.h;
     circle(c,x,y,k%18===0?1.2:.5,'rgba(204,232,249,.17)');
   }
 }
 function bindLoop(scene,render,advance,pauseId){
   const btn=document.getElementById(pauseId);
   let playing=!reduce.matches,raf=0,last=0,paint=0,localClock=0;
   function frame(now){
     raf=0;if(!playing||document.hidden||!scene.visible)return;
     const dt=last?Math.min(.060,Math.max(0,(now-last)/1000)):0;
     last=now;localClock+=dt;advance(dt,localClock);
     if(now-paint>=32){render(localClock);paint=now;}
     raf=requestAnimationFrame(frame);
   }
   function refresh(){
     if(raf)cancelAnimationFrame(raf);raf=0;last=0;paint=0;
     if(btn){btn.textContent=playing?'Pause Ⅱ':'Play ▶';btn.setAttribute('aria-label',playing?'Pause animation':'Play animation');}
     render(localClock);
     if(playing&&!document.hidden&&scene.visible)raf=requestAnimationFrame(frame);
   }
   if(btn)btn.addEventListener('click',()=>{playing=!playing;refresh();});
   if('IntersectionObserver' in window)new IntersectionObserver(e=>{scene.visible=!!e[0]&&e[0].isIntersecting;refresh();},{threshold:.015}).observe(scene.canvas);
   window.addEventListener('resize',()=>{resize(scene);render(localClock);},{passive:true});
   document.addEventListener('visibilitychange',refresh);
   if(reduce.addEventListener)reduce.addEventListener('change',()=>{if(reduce.matches)playing=false;refresh();});
   resize(scene);refresh();
   return{refresh,pause(){playing=false;refresh();},restart(){localClock=0;refresh();}};
 }

 // 01 · AGN / BLACK-HOLE DUTY CYCLE
 const agn=context('agn-scene');
 if(agn){
   let p=0,frameClock=0;
   const buttons=[...document.querySelectorAll('.astro-phase')],range=document.getElementById('agn-progress');
   const phaseLabel=document.getElementById('agn-phase'),phaseSub=document.getElementById('agn-phase-sub');
   const phases=[
     ['01 · COLD GAS SUPPLY','Fresh inflows feed the accretion disk around the black hole'],
     ['02 · AGN FEEDBACK','Accretion releases energy through radiation, winds and jets'],
     ['03 · SUPPRESSED ACCRETION','Hot, displaced gas reduces central inflow and star formation'],
     ['04 · RECOVERY & REIGNITION','Gas cools or is replenished; accretion may restart']
   ];
   const n=230,cloud=Array.from({length:n},(_,i)=>({angle:rnd()*TAU,r:.27+rnd()*.62,speed:.22+rnd()*1.4,radius:.45+rnd()*2}));
   function draw(clock){
     const c=agn.ctx,w=agn.w,h=agn.h,m=Math.min(w,h),cx=w*.51,cy=h*.51;
     base(agn);
     const exact=p*4,stage=Math.min(3,Math.floor(exact)),u=exact-stage;
     const bright=stage===0?mix(.46,1,u):stage===1?mix(1,.57,u):stage===2?mix(.57,.12,u):mix(.12,.7,u);
     const jet=stage===1?Math.sin(Math.PI*u)*.95+.17:stage===2?(1-u)*.6:0;
     const shock=stage===1?u*.82:stage===2?mix(.82,.4,u):stage===3?.35*(1-u):.06;
     const inflow=stage===0?.87:stage===1?.35:stage===2?.07:.35+.6*u;
     glow(c,cx,cy,m*.50,'26,96,149',.31);
     // Moving ambient reservoir; hot bubbles fade and cooling streams return.
     for(let k=0;k<cloud.length;k++){
       const q=cloud[k],rad=m*(.16+q.r*.63),phi=q.angle+clock*.14*q.speed;
       const x=cx+Math.cos(phi)*rad*1.12,y=cy+Math.sin(phi)*rad*.61;
       const warmed=stage===1||stage===2;
       circle(c,x,y,q.radius*(.55+.7*inflow),
          warmed?'rgba(237,120,94,'+(.13+.22*shock)+')':'rgba(112,209,255,'+(.15+.32*inflow)+')');
     }
     if(shock>.03){
       for(let ring=0;ring<3;ring++){
         const rr=m*(.19+.27*shock+ring*.07);
         c.beginPath();c.ellipse(cx,cy,rr,rr*.67,0,0,TAU);
         c.lineWidth=1.2;c.strokeStyle='rgba(255,123,83,'+(.27*shock*(1-ring*.22))+')';c.stroke();
       }
       glow(c,cx,cy,m*(.28+.3*shock),'218,83,70',.18*shock);
     }
     // Spiral inflows are bright during feeding/recovery, suppressed by outflow.
     c.save();c.globalCompositeOperation='screen';
     for(let arm=0;arm<5;arm++){
       c.beginPath();
       for(let j=0;j<130;j++){
         const t=j/129,r=m*(.17+t*.43),a=arm*TAU/5+clock*(.32+.27*inflow)+2.8*Math.log(.18+t);
         const x=cx+Math.cos(a)*r,y=cy+Math.sin(a)*r*.51;
         if(j===0)c.moveTo(x,y);else c.lineTo(x,y);
       }
       c.strokeStyle='rgba(104,207,252,'+(.06+.45*inflow)+')';c.lineWidth=1.3;c.stroke();
     }
     // The accretion disk has a dark centre and differential rotation.
     for(let ring=0;ring<19;ring++){
       const r=m*(.065+ring*.0083),a=(ring/19)*TAU+clock*mix(1.2,.35,ring/19);
       c.beginPath();c.ellipse(cx,cy,r,r*.31,-.16,0,TAU);
       c.strokeStyle='rgba('+(ring<9?'255,186,119':'92,201,236')+','+(.18+.5*bright*(1-ring/23))+')';
       c.lineWidth=ring<8?2.3:1.2;c.stroke();
       let px=cx+Math.cos(a)*r,py=cy+Math.sin(a)*r*.31;
       circle(c,px,py,.65+bright*.65,'rgba(255,214,172,'+(.3+bright*.5)+')');
     }
     glow(c,cx,cy,m*.13,'255,170,102',.27*bright);
     // Central BH shadow, a stylized luminous crescent.
     circle(c,cx,cy,m*.047,'#000207');
     c.beginPath();c.arc(cx,cy,m*.050,.18,Math.PI*1.72);
     c.strokeStyle='rgba(255,210,145,'+(.4+.55*bright)+')';c.lineWidth=3;c.stroke();
     // Bipolar jets emerge during the active feedback phase.
     if(jet>.02){
       for(const sign of [-1,1]){
         const len=m*(.17+.48*jet),end=cy+sign*len;
         const g=c.createLinearGradient(cx,cy,cx,end);
         g.addColorStop(0,'rgba(180,245,255,'+(.72*jet)+')');
         g.addColorStop(.7,'rgba(83,153,255,'+(.32*jet)+')');
         g.addColorStop(1,'rgba(69,136,241,0)');
         c.beginPath();c.moveTo(cx-8,cy);c.quadraticCurveTo(cx-20,cy+sign*len*.65,cx-38*jet,end);
         c.lineTo(cx+38*jet,end);c.quadraticCurveTo(cx+20,cy+sign*len*.65,cx+8,cy);c.closePath();
         c.fillStyle=g;c.fill();
         stroke(c,cx,cy,cx+Math.sin(clock*.4+sign)*5,end,'rgba(207,248,255,'+(.55*jet)+')',2.2);
         glow(c,cx,end,m*.055,'75,150,242',jet*.19);
       }
     }
     // Cooling flows reappear after the previous feedback phase.
     if(stage===3){
       const f=u;
       for(let k=0;k<40;k++){
         const a=k*2.399+clock*.18,rad=m*(.55-.39*f*(k%7)/7);
         const x=cx+Math.cos(a)*rad,y=cy+Math.sin(a)*rad*.67;
         circle(c,x,y,1.0+1.4*f,'rgba(77,202,232,'+(.25+.54*f)+')');
       }
     }
     // UI text is in a high-contrast HTML overlay, not baked into the canvas.
     if(phaseLabel)phaseLabel.textContent=phases[stage][0];
     if(phaseSub)phaseSub.textContent=phases[stage][1];
     buttons.forEach(b=>{const a=Number(b.dataset.agnPhase)===stage;b.classList.toggle('active',a);b.setAttribute('aria-pressed',String(a));});
     if(range&&document.activeElement!==range)range.value=String(Math.round(p*1000));
   }
   const loop=bindLoop(agn,draw,dt=>{p=(p+dt/28)%1;},'agn-play');
   buttons.forEach(b=>b.addEventListener('click',()=>{p=(Number(b.dataset.agnPhase)+.02)/4;loop.pause();}));
   if(range)range.addEventListener('input',()=>{p=clamp(Number(range.value)/1000,0,.9999);loop.pause();});
 }

 // The exact original Pop III.1 MP4 is embedded unchanged via Google Drive.
 // No schematic animation is substituted for the supplied footage.
})();

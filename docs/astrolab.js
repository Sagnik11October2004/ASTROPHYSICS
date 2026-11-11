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

 // 02 · POP III.1: four-stage comparison based on user's 15-second source video.
 const pop=context('popiii-scene');
 if(pop){
   let p=0;
   const range=document.getElementById('pop-progress'),buttons=[...document.querySelectorAll('.pop-phase-btn')];
   const title=document.getElementById('pop-phase'),description=document.getElementById('pop-description');
   const stages=[
     ['STAGE 1 · PROTOSTAR FORMATION','Both cases begin with primordial gas accretion inside a minihalo.'],
     ['STAGE 2 · DIVERGENT EVOLUTION','Without extra heating the star contracts; annihilation heating can maintain a larger radius.'],
     ['STAGE 3 · UV FEEDBACK & ACCRETION','A compact hot star more strongly ionizes its inflow; an inflated cool star may continue accreting.'],
     ['STAGE 4 · POSSIBLE LATER GROWTH','If dark-matter heating persists, delayed feedback might allow a more massive object or black-hole seed.']
   ];
   const gas=Array.from({length:260},()=>({angle:rnd()*TAU,r:rnd(),size:.35+rnd()*1.4,speed:.18+rnd()*.8,phase:rnd()*TAU}));
   function draw(clock){
     const c=pop.ctx,w=pop.w,h=pop.h,m=Math.min(w/2,h),split=w*.5;
     base(pop);
     const idx=Math.min(3,Math.floor(p*4)),u=p*4-idx,step=idx+u;
     c.fillStyle='rgba(6,20,38,.58)';c.fillRect(0,0,split,h);
     c.fillStyle='rgba(22,17,34,.62)';c.fillRect(split,0,split,h);
     stroke(c,split,0,split,h,'rgba(199,229,235,.30)',1.3);
     for(let side=0;side<2;side++){
       const x=split*(side+.5),y=h*.51;
       const heating=side===1;
       const radius0=[27,22,17,15,13],radius1=[22,27,35,43,55];
       const R=(heating?mix(radius1[idx],radius1[idx+1],u):mix(radius0[idx],radius0[idx+1],u))*(m/430);
       const Feedback=heating?.14+.14*smooth((step-1.8)/1.4):.15+.85*smooth((step-1.05)/1.6);
       const Accretion=heating?.75+.20*smooth((step-2)/2):1-.86*smooth((step-1.2)/1.6);
       const UV=heating?.11+.14*smooth((step-2)/2):.16+.8*smooth((step-1)/1.4);
       const dm=heating?.28+.64*smooth(step/3):0;
       glow(c,x,y,m*.40,heating?'152,86,176':'52,90,163',.16);
       // A changing annihilation-heating cusp surrounds the right-hand core.
       if(heating){
         for(let ring=0;ring<5;ring++){
           const rr=R*(1.48+ring*.45);
           c.beginPath();c.arc(x,y,rr,0,TAU);
           c.lineWidth=1.3;c.strokeStyle='rgba(196,125,226,'+(.03+dm*.12/(ring+1))+')';c.stroke();
         }
       }
       // Inflow particles spiral in; heating may postpone strong feedback.
       for(let k=0;k<gas.length;k++){
         const q=gas[k],r=m*(.11+q.r*.40);
         const a=q.angle+clock*q.speed*.5;
         const spiral=1-.36*Accretion*((clock*.1+q.phase)%1);
         const X=x+Math.cos(a)*r*spiral,Y=y+Math.sin(a)*r*spiral;
         const alpha=(.18+.5*Accretion)*(1-Feedback*.32);
         circle(c,X,Y,q.size*(.55+Accretion*.65),heating?'rgba(255,192,136,'+alpha+')':'rgba(134,207,255,'+alpha+')');
       }
       for(let arm=0;arm<6;arm++){
         c.beginPath();
         for(let j=0;j<110;j++){
           const t=j/109,r=m*(.12+t*.37),a=arm*TAU/6+clock*.23+2.1*Math.log(t+.2);
           const X=x+Math.cos(a)*r,Y=y+Math.sin(a)*r;
           if(j===0)c.moveTo(X,Y);else c.lineTo(X,Y);
         }
         c.strokeStyle=heating?'rgba(249,179,112,'+(.11+.23*Accretion)+')':'rgba(124,201,245,'+(.11+.18*Accretion)+')';
         c.lineWidth=1;c.stroke();
       }
       if(UV>.1){
         // Ionizing UV pressure / H II region (schematic, not a Strömgren solution).
         const rad=(R*(2+3*UV));
         for(let ring=0;ring<2;ring++){
           c.beginPath();c.arc(x,y,rad*(1+.4*ring),0,TAU);
           c.lineWidth=1.4;c.strokeStyle='rgba(112,194,255,'+((.12+.36*UV)*(1-ring*.45))+')';c.stroke();
         }
         for(let ray=0;ray<14;ray++){
           const a=ray*TAU/14+clock*.015;
           const r0=R*1.3,r1=R*(1.8+UV*4.2);
           stroke(c,x+Math.cos(a)*r0,y+Math.sin(a)*r0,x+Math.cos(a)*r1,y+Math.sin(a)*r1,'rgba(132,214,255,'+(.08+.5*UV)+')',.65);
         }
       }
       glow(c,x,y,R*(heating?5.6:4),heating?'255,153,74':'169,205,255',heating?.29:.29+.3*UV);
       const starColor=heating?'255,199,127':step>1?'220,241,255':'255,204,119';
       const stellar=c.createRadialGradient(x-R*.35,y-R*.38,R*.1,x,y,R);
       stellar.addColorStop(0,'#fff9d5');stellar.addColorStop(.55,heating?'#ffca80':step>1?'#c7e6ff':'#f9ba73');
       stellar.addColorStop(1,heating?'#ec813a':'#96bcf6');
       circle(c,x,y,R,stellar);
       c.beginPath();c.arc(x,y,R*1.07,0,TAU);
       c.strokeStyle=heating?'rgba(255,188,104,.65)':'rgba(180,222,255,.62)';c.lineWidth=1.4;c.stroke();
       // Simple gas flow metric rather than invented stellar masses.
       c.save();c.fillStyle='rgba(6,19,34,.84)';c.strokeStyle='rgba(181,218,233,.27)';
       const bx=side===0?w*.035:split+w*.035,by=h*.80,bw=split*.66,bh=h*.135;
       c.fillRect(bx,by,bw,bh);c.strokeRect(bx,by,bw,bh);
       c.textAlign='left';c.font='600 '+Math.round(Math.max(10,m*.027))+'px system-ui';c.fillStyle='#d8e9f3';
       c.fillText(heating?'DM heating: illustrative ON':'DM heating: OFF',bx+12,by+21);
       c.fillStyle='#a9c6d7';c.font=Math.round(Math.max(10,m*.024))+'px system-ui';
       c.fillText('Relative accretion supply',bx+12,by+42);
       const meterx=bx+12,metery=by+bh-16,meterw=bw-24;
       c.fillStyle='rgba(115,154,177,.2)';c.fillRect(meterx,metery,meterw,6);
       c.fillStyle=heating?'#ffbd7c':'#8ad5f2';c.fillRect(meterx,metery,meterw*Accretion,6);
       c.restore();
     }
     if(title)title.textContent=stages[idx][0];
     if(description)description.textContent=stages[idx][1];
     buttons.forEach(b=>{const active=Number(b.dataset.popPhase)===idx;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
     if(range&&document.activeElement!==range)range.value=String(Math.round(p*1000));
   }
   const loop=bindLoop(pop,draw,dt=>{p=(p+dt/30)%1;},'pop-play');
   buttons.forEach(b=>b.addEventListener('click',()=>{p=(Number(b.dataset.popPhase)+.02)/4;loop.pause();}));
   if(range)range.addEventListener('input',()=>{p=clamp(Number(range.value)/1000,0,.9999);loop.pause();});
 }

 // Detailed browser Monte Carlo demonstration is now in mcrt-source.js,
 // adapted from the original private Nordita C scattering and transport methods.
})();

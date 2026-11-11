'use strict';
/* Six-stage cosmological visualization. Redshift anchors are chronological:
   the CMB at z~1100 precedes the user's requested z=499 start.
   The matter evolution is a genuine 2D first-order Lagrangian displacement
   visualization using a Fourier potential, NOT a 3D LCDM N-body or hydro run.
   Star/halo/galaxy icons, light cones and the CMB sphere are illustration layers. */
(() => {
 const canvas=document.getElementById('cosmos-story-canvas');
 if(!canvas || !canvas.getContext)return;
 const c=canvas.getContext('2d',{alpha:false,desynchronized:true});
 if(!c)return;
 const slider=document.getElementById('cosmo-progress');
 const play=document.getElementById('cosmo-play');
 const redshift=document.getElementById('cosmo-redshift');
 const nameOut=document.getElementById('cosmo-epoch-readout');
 const names=document.getElementById('story-name'),sub=document.getElementById('story-subtitle'),number=document.getElementById('story-number');
 const chapterNodes=[...document.querySelectorAll('.epoch')];
 const reduce=window.matchMedia('(prefers-reduced-motion: reduce)');
 let p=reduce.matches?.64:.012,playing=!reduce.matches,raf=0,visible=true,last=0,clock=0,paint=0,W=800,H=600,DPR=1;
 let seed=990499;function rnd(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
 const TAU=Math.PI*2,clamp=(v,a,b)=>Math.max(a,Math.min(b,v)),mix=(a,b,t)=>a+(b-a)*t,smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
 const S=[
   {title:'CMB · LAST SCATTERING',sub:'Primordial temperature fluctuations, before z=499',z:1100},
   {title:'INITIAL CONDITIONS · DARK AGES',sub:'Nearly uniform primordial matter at z=499',z:499},
   {title:'MINIHALO COLLAPSE',sub:'Rare, high-density dark-matter haloes assemble first',z:35},
   {title:'FIRST STARS · COSMIC DAWN',sub:'Primordial gas collapses within minihaloes',z:22},
   {title:'GALAXIES & REIONIZATION',sub:'Luminous galaxies grow within dark-matter haloes',z:6},
   {title:'THE UNIVERSE TODAY',sub:'A rich cosmic web of filaments, groups and clusters',z:0}
 ];
 const stageStops=[0,.085,.225,.41,.60,.80,1];
 const zAnchor=[1100,499,35,22,6,2,0];
 function getState(){
   let chapter=5;
   for(let i=0;i<6;i++){if(p<stageStops[i+1]){chapter=i;break;}}
   const u=clamp((p-stageStops[chapter])/(stageStops[chapter+1]-stageStops[chapter]),0,1);
   const z=Math.exp(mix(Math.log(1+zAnchor[chapter]),Math.log(1+zAnchor[chapter+1]),u))-1;
   return {chapter,u,z,growth:clamp(1.2*Math.pow(1/(1+z),.55),0,1.2)};
 }
 const stars=Array.from({length:225},()=>({x:rnd(),y:rnd(),r:.3+rnd(),tw:rnd()*TAU}));
 const modes=Array.from({length:41},()=>{
   let kx=Math.floor(rnd()*11)-5,ky=Math.floor(rnd()*11)-5;
   if(!kx&&!ky)kx=1;
   const k2=kx*kx+ky*ky;
   return {kx,ky,phase:rnd()*TAU,amp:(.6+rnd())/Math.pow(k2,1.55)};
 });
 // Displacements follow psi=-grad(phi) for a Fourier-space potential field.
 function disp(x,y){
   let dx=0,dy=0;
   for(const m of modes){
     const amp=m.amp*Math.sin(TAU*(m.kx*x+m.ky*y)+m.phase);
     dx+=m.kx*amp;dy+=m.ky*amp;
   }
   return [dx,dy];
 }
 const N=4500;const Q=new Float32Array(N*4),particles=[];
 let maximum=0;
 for(let i=0;i<N;i++){
   const x=(i%90+.3+rnd()*.4)/90, y=(Math.floor(i/90)+.3+rnd()*.4)/50;
   const v=disp(x,y);maximum=Math.max(maximum,Math.hypot(...v));
   Q[i*4]=x;Q[i*4+1]=y;Q[i*4+2]=v[0];Q[i*4+3]=v[1];
 }
 for(let i=0;i<N;i++){Q[i*4+2]/=maximum||1;Q[i*4+3]/=maximum||1;}
 const haloSeeds=Array.from({length:29},(_,i)=>{
   let x=.09+rnd()*.82,y=.11+rnd()*.78;
   return{x,y,weight:.5+rnd(),delay:rnd()*.3,spin:rnd()*TAU};
 });
 const filaments=[];
 for(let i=0;i<haloSeeds.length;i++){
   const a=haloSeeds[i], near=haloSeeds.map((b,j)=>({j,d:Math.hypot(a.x-b.x,a.y-b.y)})).filter(o=>o.j!==i).sort((a,b)=>a.d-b.d).slice(0,3);
   for(const entry of near){if(i<entry.j&&!filaments.some(e=>e[0]===i&&e[1]===entry.j))filaments.push([i,entry.j]);}
 }
 function circle(x,y,r,color){c.beginPath();c.arc(x,y,Math.max(.2,r),0,TAU);c.fillStyle=color;c.fill();}
 function line(x1,y1,x2,y2,color,width=1){c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.strokeStyle=color;c.lineWidth=width;c.stroke();}
 function glow(x,y,R,rgb,amp){
   const g=c.createRadialGradient(x,y,0,x,y,R);
   g.addColorStop(0,'rgba('+rgb+','+amp+')');g.addColorStop(.45,'rgba('+rgb+','+(amp*.3)+')');g.addColorStop(1,'rgba('+rgb+',0)');
   c.fillStyle=g;c.fillRect(x-R,y-R,R*2,R*2);
 }
 function photostar(x,y,r,color='255,229,167',brightness=1){
   glow(x,y,r*16,color,.32*brightness);
   circle(x,y,r,'rgba('+color+','+clamp(.7+brightness*.26,0,1)+')');
   if(r>1.1){line(x-r*4,y,x+r*4,y,'rgba('+color+','+(.23*brightness)+')',.7);line(x,y-r*4,x,y+r*4,'rgba('+color+','+(.2*brightness)+')',.7);}
 }
 function background(){
   const g=c.createLinearGradient(0,0,W,H);g.addColorStop(0,'#020612');g.addColorStop(.5,'#071426');g.addColorStop(1,'#01050f');
   c.fillStyle=g;c.fillRect(0,0,W,H);
   for(const v of stars){
     const a=.18+.21*(.5+.5*Math.sin(clock*.45+v.tw));
     circle(v.x*W,v.y*H,v.r,'rgba(206,226,246,'+a+')');
   }
 }
 function cmb(t){
   const R=Math.min(W,H)*.37,x=W*.55,y=H*.49;
   glow(x,y,R*1.8,'185,104,80',.26);
   const disk=c.createRadialGradient(x-R*.25,y-R*.2,R*.08,x,y,R*1.14);
   disk.addColorStop(0,'#ffdea7');disk.addColorStop(.20,'#ed995d');disk.addColorStop(.55,'#476a9e');disk.addColorStop(.85,'#1d3f70');disk.addColorStop(1,'rgba(16,39,77,0)');
   c.fillStyle=disk;circle(x,y,R,c.fillStyle);
   c.save();c.beginPath();c.arc(x,y,R*.97,0,TAU);c.clip();
   for(let k=0;k<320;k++){
     const a=stars[k%stars.length].tw*12+clock*.035;
     const r=Math.sqrt((k*.617033)%1)*R*.92;
     const xx=x+Math.cos(a)*r,yy=y+Math.sin(a)*r;
     glow(xx,yy,R*(.05+.1*((k*7)%17)/17),k%2?'235,105,71':'66,116,218',.12+.045*Math.sin(clock+k));
   }
   c.restore();
   for(let k=0;k<6;k++){
     c.beginPath();c.ellipse(x,y,R*(.27+k*.15),R*(.18+k*.15),clock*.008,0,TAU);
     c.strokeStyle='rgba(255,229,161,'+(.09-k*.011)+')';c.lineWidth=1;c.stroke();
   }
   c.beginPath();c.arc(x,y,R,0,TAU);c.strokeStyle='rgba(211,225,252,.25)';c.lineWidth=1.5;c.stroke();
   // Fade the last-scattering sky toward the post-CMB IC field.
   c.fillStyle='rgba(2,8,20,'+(.25*smooth(t))+')';c.fillRect(0,0,W,H);
 }
 function field(st){
   const alpha=st.chapter===1?.52:1;
   const g=st.growth;
   // Cosmic filament density is drawn from evolving Zel'dovich particle positions.
   c.save();c.globalCompositeOperation='screen';
   for(let i=0;i<N;i++){
     const x=Q[i*4],y=Q[i*4+1],dx=Q[i*4+2],dy=Q[i*4+3];
     const px=(x+g*dx*.24)%1,py=(y+g*dy*.24)%1;
     const X=(px<0?px+1:px)*W,Y=(py<0?py+1:py)*H;
     const tint=st.chapter<2?'126,157,194':st.chapter<4?'116,166,222':'94,194,245';
     const opacity=alpha*(.12+.17*g);
     c.fillStyle='rgba('+tint+','+opacity+')';
     c.fillRect(X,Y,st.chapter<3?.7:1,st.chapter<3?.7:1);
   }
   c.restore();
   const fade=st.chapter<2?0:smooth((p-.20)/.46);
   if(fade>0){
     // Coarse nonlinear morphology overlay: haloes/filaments are illustrative.
     for(let i=0;i<filaments.length;i++){
       const [aIndex,bIndex]=filaments[i];const a=haloSeeds[aIndex],b=haloSeeds[bIndex];
       const x1=a.x*W,y1=a.y*H,x2=b.x*W,y2=b.y*H;
       const bend=Math.sin((aIndex+bIndex)*2.1+clock*.1)*Math.min(W,H)*.045;
       c.beginPath();c.moveTo(x1,y1);c.quadraticCurveTo((x1+x2)/2+bend,(y1+y2)/2-bend,x2,y2);
       c.strokeStyle='rgba(43,118,191,'+(.07+.22*fade)+')';c.lineWidth=4;c.stroke();
       c.strokeStyle='rgba(114,211,255,'+(.08+.3*fade)+')';c.lineWidth=.7;c.stroke();
     }
     for(let i=0;i<haloSeeds.length;i++){
       const o=haloSeeds[i],x=o.x*W,y=o.y*H,r=Math.min(W,H)*(.035+.045*o.weight)*fade;
       glow(x,y,r*3,'57,131,204',.11*fade);
       c.beginPath();c.arc(x,y,r,0,TAU);c.strokeStyle='rgba(81,185,225,'+(.05+.3*fade)+')';c.lineWidth=1.4;c.stroke();
     }
   }
 }
 function dawn(st){
   if(st.chapter<3)return;
   const born=smooth((p-.405)/.36);
   const count=st.chapter===3?Math.floor(2+16*st.u):Math.floor(18+56*born);
   for(let i=0;i<count;i++){
     const n=haloSeeds[(i*7)%haloSeeds.length];
     const t=clock*.09+i*2.6,dist=Math.min(W,H)*(.013+(i%9)*.004);
     const x=n.x*W+Math.sin(t)*dist,y=n.y*H+Math.cos(t*1.2)*dist;
     const r=.65+(i%4)*.35;
     photostar(x,y,r,i%4===0?'185,214,255':'255,218,158',born*.9);
   }
 }
 function galaxies(st){
   if(st.chapter<4)return;
   const f=smooth((p-.57)/.32);
   const count=Math.floor(4+26*f);
   for(let i=0;i<count;i++){
     const n=haloSeeds[i%haloSeeds.length],x=n.x*W,y=n.y*H;
     const R=(3+(i%4)*1.5)*(1+f*2.8);
     glow(x,y,R*3,i%4===0?'194,218,255':'124,180,244',.3+.15*f);
     c.save();c.translate(x,y);c.rotate(n.spin+clock*.055);
     for(let arm=0;arm<2;arm++){
       c.beginPath();
       for(let j=0;j<35;j++){
         const theta=j*.13+arm*Math.PI,rr=(1+j*.045)*R*.3;
         const xx=Math.cos(theta)*rr,yy=Math.sin(theta)*rr*.5;
         if(j===0)c.moveTo(xx,yy);else c.lineTo(xx,yy);
       }
       c.strokeStyle='rgba(212,224,255,'+(.3+.35*f)+')';c.lineWidth=1.4;c.stroke();
     }
     c.restore();
     photostar(x,y,.8+1.3*f,'255,235,189',.8);
   }
 }
 function scene(){
   const st=getState();background();
   if(st.chapter===0)cmb(st.u);
   else{
     const ambient=st.chapter===1?.10:st.chapter===2?.22:.32;
     glow(W*.55,H*.52,Math.max(W,H)*.8,'34,80,148',ambient);
     field(st);
     dawn(st);
     galaxies(st);
     if(st.chapter===1){
       // Modest primordial fluctuations at z=499; no visible stellar sources.
       const f=smooth(st.u);
       for(let i=0;i<8;i++)glow(W*(.17+i*.11),H*(.25+.3*Math.sin(i)),Math.min(W,H)*(.05+f*.05),'60,98,155',.07+.06*f);
     }
     if(st.chapter===3)for(let i=0;i<4;i++)glow(haloSeeds[i].x*W,haloSeeds[i].y*H,Math.min(W,H)*.15,'243,161,87',.13*st.u);
   }
   const v=c.createRadialGradient(W*.52,H*.50,Math.min(W,H)*.08,W*.52,H*.5,Math.max(W,H)*.75);
   v.addColorStop(0,'rgba(0,0,0,0)');v.addColorStop(1,'rgba(0,2,9,.65)');
   c.fillStyle=v;c.fillRect(0,0,W,H);
   const zStr=st.chapter===0?'z ≈ 1100 · CMB prologue':'z ≈ '+(st.z>=100?Math.round(st.z):st.z>=1?st.z.toFixed(1):st.z.toFixed(2));
   // The illustration's scientific redshift and epoch labels must NEVER be
   // overlaid on externally hosted videos with different simulation ranges.
   if(document.body.classList.contains('cosmo-model-active')){
     if(redshift)redshift.textContent=zStr;
     if(nameOut)nameOut.textContent=S[st.chapter].title;
     if(names)names.textContent=S[st.chapter].title;
     if(sub)sub.textContent=S[st.chapter].sub;
     if(number)number.textContent=String(st.chapter+1).padStart(2,'0')+' / 06';
     chapterNodes.forEach((n,i)=>n.classList.toggle('current-epoch',i===st.chapter));
     if(slider&&document.activeElement!==slider)slider.value=String(Math.round(p*1000));
   }
 }
 function size(){
   const rect=canvas.getBoundingClientRect();W=Math.max(200,rect.width);H=Math.max(180,rect.height);DPR=Math.min(window.devicePixelRatio||1,1.5);
   canvas.width=Math.round(W*DPR);canvas.height=Math.round(H*DPR);c.setTransform(DPR,0,0,DPR,0,0);scene();
 }
 function loop(now){
   raf=0;if(!playing||document.hidden||!visible)return;
   let dt=last?Math.max(0,Math.min(.06,(now-last)/1000)):0;clock+=dt;last=now;p=(p+dt/31)%1;
   if(now-paint>35){scene();paint=now;}
   raf=requestAnimationFrame(loop);
 }
 function sync(){
   if(raf)cancelAnimationFrame(raf);raf=0;last=0;
   if(play){play.textContent=playing?'Pause Ⅱ':'Play ▶';play.setAttribute('aria-label',playing?'Pause cosmology animation':'Play cosmology animation');}
   scene();if(playing&&!document.hidden&&visible)raf=requestAnimationFrame(loop);
 }
 if(play)play.addEventListener('click',()=>{playing=!playing;sync();});
 if(slider)slider.addEventListener('input',()=>{playing=false;p=clamp(Number(slider.value)/1000,0,.999999);sync();});
 if('IntersectionObserver' in window)new IntersectionObserver(x=>{visible=!!x[0]&&x[0].isIntersecting;sync();},{threshold:.01}).observe(canvas);
 window.addEventListener('resize',size,{passive:true});document.addEventListener('visibilitychange',sync);
 if(reduce.addEventListener)reduce.addEventListener('change',()=>{if(reduce.matches)playing=false;sync();});
 size();sync();
})();

'use strict';

/* Visible, continuously evolving five-epoch cosmic-history animation.
   Later structure growth uses a genuine 2D first-order displacement model:
   x(q,D) = q + D * psi(q), with psi from Fourier potential modes.
   Plasma, CMB texture, first-star ignition and bright galaxies are
   intentionally stylized scientific illustrations, not a 3D N-body
   / hydrodynamics / radiative-transfer calculation. No external assets. */
(function () {
  const canvas=document.getElementById('cosmos-story-canvas');
  const play=document.getElementById('cosmo-play');
  const slider=document.getElementById('cosmo-progress');
  const redshift=document.getElementById('cosmo-redshift');
  const epoch=document.getElementById('cosmo-epoch-readout');
  const stageNumber=document.getElementById('story-number');
  const stageName=document.getElementById('story-name');
  const stageSubtitle=document.getElementById('story-subtitle');
  const epochNodes=Array.from(document.querySelectorAll('.epoch'));
  if(!canvas||!canvas.getContext)return;
  const ctx=canvas.getContext('2d',{alpha:false,desynchronized:true});
  if(!ctx)return;
  const reduced=window.matchMedia('(prefers-reduced-motion:reduce)');
  const labels=[
    {name:'HOT EARLY UNIVERSE',sub:'A hot, dense primordial plasma',z:'z ≫ 1100'},
    {name:'RECOMBINATION & CMB',sub:'The universe becomes transparent',z:'z ≈ 1100'},
    {name:'COSMIC DARK AGES',sub:'Neutral hydrogen before the first stars',z:'z ≈ 1100–30'},
    {name:'COSMIC DAWN',sub:'The earliest stars illuminate their surroundings',z:'z ≈ 30–10'},
    {name:'THE COSMIC WEB',sub:'Galaxies and dark matter form large-scale structure',z:'z ≈ 10–0'}
  ];
  let w=800,h=620,dpr=1,playing=!reduced.matches,p=reduced.matches?.88:.035;
  let raf=0,last=0,visible=true,elapsed=0,needsRender=true;
  let seed=97131;
  function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
  const cl=(x,a,b)=>Math.min(b,Math.max(a,x));
  const mix=(a,b,t)=>a+(b-a)*t;
  const fract=x=>x-Math.floor(x);
  const pts=Array.from({length:1400},()=>({x:random(),y:random(),r:.45+random()*.85,phase:random()*6.283}));
  const noise=Array.from({length:190},()=>({x:random(),y:random(),r:1.2+random()*4.0,phi:random()*Math.PI*2}));
  const nodes=Array.from({length:24},(_,i)=>{
    const angle=i*2.3999632,rad=.085+.36*Math.sqrt((i+.5)/24);
    return {x:.52+Math.cos(angle)*rad*.95,y:.49+Math.sin(angle)*rad*.75,phase:random()*Math.PI*2,size:2+random()*3};
  });
  const edges=[];
  nodes.forEach((a,i)=>{
    const nearest=nodes.map((b,j)=>({j,d:Math.hypot(b.x-a.x,b.y-a.y)}))
      .filter(o=>o.j!==i).sort((a,b)=>a.d-b.d).slice(0,3);
    nearest.forEach(o=>{if(i<o.j&&!edges.some(e=>e.i===i&&e.j===o.j))edges.push({i,j:o.j,phase:random()*9});});
  });
  // First-order Fourier displacement field: psi = -grad(phi).
  const modes=Array.from({length:29},()=>{
    let kx=Math.floor(random()*11)-5,ky=Math.floor(random()*11)-5;
    if(kx===0&&ky===0)kx=1;
    const k2=kx*kx+ky*ky;
    return {kx,ky,amp:(.65+random())/Math.pow(k2,1.5),phase:random()*Math.PI*2};
  });
  const qx=new Float32Array(pts.length),qy=new Float32Array(pts.length);
  const vx=new Float32Array(pts.length),vy=new Float32Array(pts.length);
  let peak=0;
  pts.forEach((v,i)=>{
    qx[i]=v.x;qy[i]=v.y;
    let dx=0,dy=0;
    modes.forEach(m=>{
      const c=Math.sin(6.2831853*(m.kx*v.x+m.ky*v.y)+m.phase)*m.amp;
      dx+=c*m.kx;dy+=c*m.ky;
    });
    vx[i]=dx;vy[i]=dy;peak=Math.max(peak,Math.hypot(dx,dy));
  });
  for(let i=0;i<pts.length;i++){vx[i]/=(peak||1);vy[i]/=(peak||1);}
  const clampAlpha=x=>cl(x,0,1);
  function gradient(x,y,r,colors){
    const g=ctx.createRadialGradient(x,y,0,x,y,r);
    for(const [p,c] of colors)g.addColorStop(p,c);
    return g;
  }
  function disc(x,y,r,fill){
    ctx.beginPath();ctx.arc(x,y,Math.max(.2,r),0,Math.PI*2);
    ctx.fillStyle=fill;ctx.fill();
  }
  function line(x1,y1,x2,y2,color,lw){
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
    ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.stroke();
  }
  function nebula(x,y,r,rgb,a){
    ctx.fillStyle=gradient(x,y,r,[[0,'rgba('+rgb+','+a+')'],[.35,'rgba('+rgb+','+(a*.32)+')'],[1,'rgba('+rgb+',0)']]);
    ctx.fillRect(x-r,y-r,r*2,r*2);
  }
  function star(x,y,r,intensity,col){
    const g=ctx.createRadialGradient(x,y,0,x,y,r*8);
    g.addColorStop(0,'rgba('+col+','+clampAlpha(intensity)+')');
    g.addColorStop(.17,'rgba('+col+','+clampAlpha(intensity*.47)+')');
    g.addColorStop(1,'rgba('+col+',0)');
    ctx.fillStyle=g;ctx.fillRect(x-r*8,y-r*8,r*16,r*16);
    disc(x,y,r, 'rgba(252,251,235,'+clampAlpha(.5+intensity*.4)+')');
    if(r>1.5){
      line(x-r*5,y,x+r*5,y,'rgba('+col+','+clampAlpha(intensity*.18)+')',.8);
      line(x,y-r*5,x,y+r*5,'rgba('+col+','+clampAlpha(intensity*.14)+')',.8);
    }
  }
  function base(){
    const bg=ctx.createLinearGradient(0,0,w,h);
    bg.addColorStop(0,'#02050d');bg.addColorStop(.52,'#071221');bg.addColorStop(1,'#020711');
    ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
    // A persistent subtle starfield anchors the image throughout the timeline.
    noise.forEach((s,i)=>{
      const a=.15+.22*Math.abs(Math.sin(s.phi+elapsed*.17));
      ctx.fillStyle='rgba(187,217,239,'+a+')';
      ctx.fillRect(s.x*w,s.y*h,i%17===0?1.8:.8,i%17===0?1.8:.8);
    });
  }
  function hot(t){
    const cx=w*.53,cy=h*.49,R=Math.min(w,h)*(.31+.075*t);
    nebula(cx,cy,R*2.1,'230,68,33',.30);
    ctx.fillStyle=gradient(cx,cy,R*1.38,[
      [0,'rgba(255,252,205,.99)'],[.14,'rgba(255,210,99,.97)'],
      [.42,'rgba(243,110,49,.77)'],[.68,'rgba(178,48,60,.42)'],[1,'rgba(147,29,58,0)']
    ]);
    disc(cx,cy,R*1.38,ctx.fillStyle);
    for(let k=0;k<9;k++){
      const radius=R*(.28+k*.16)+Math.sin(elapsed*2.1+k)*R*.03;
      ctx.beginPath();ctx.ellipse(cx,cy,radius,radius*(.78+.02*k),-0.25,0,Math.PI*2);
      ctx.strokeStyle='rgba(255,196,111,'+(.17-k*.014)+')';ctx.lineWidth=1.3;ctx.stroke();
    }
    for(let i=0;i<pts.length;i+=3){
      const a=pts[i].x*6.283+elapsed*.20,rr=R*(.16+pts[i].y*1.8)+t*R*.28;
      const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr*.86;
      disc(x,y,.5+pts[i].r*.6,'rgba(255,205,139,'+(.18+.5*pts[i].x)+')');
    }
  }
  function cmb(t){
    const cx=w*.53,cy=h*.49,r=Math.min(w,h)*.38;
    nebula(cx,cy,r*2,'62,109,177',.25);
    ctx.save();ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.clip();
    ctx.fillStyle=gradient(cx-r*.21,cy-r*.18,r*1.3,[
      [0,'rgba(234,195,134,1)'],[.35,'rgba(189,111,80,1)'],
      [.7,'rgba(55,112,174,1)'],[1,'rgba(17,58,112,1)']
    ]);ctx.fillRect(cx-r,cy-r,2*r,2*r);
    for(let k=0;k<110;k++){
      const a=noise[k%noise.length].x*6.283*3,
        rr=Math.sqrt(noise[k%noise.length].y)*r*.95;
      const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr;
      nebula(x,y,r*(.08+.16*noise[k].r/5),
        k%3===0?'226,116,76':k%3===1?'48,117,185':'250,209,143',.20);
    }
    for(let j=0;j<12;j++){
      const a=j*.83+elapsed*.045;
      line(cx+Math.cos(a)*r*.45,cy+Math.sin(a)*r*.45,
           cx+Math.cos(a)*r*.98,cy+Math.sin(a)*r*.98,'rgba(250,231,179,.11)',1);
    }
    ctx.restore();
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle='rgba(228,218,193,.38)';ctx.lineWidth=1.7;ctx.stroke();
    // Transparency from last scattering transitions the scene toward darkness.
    nebula(cx,cy,r*1.3,'113,137,199',.14);
  }
  function darkAges(t){
    const cx=w*.52,cy=h*.49,R=Math.min(w,h)*.68;
    nebula(cx,cy,R,'18,45,89',.48);
    nebula(w*.70,h*.34,R*.46,'51,28,91',.22);
    for(let i=0;i<pts.length;i+=2){
      const v=pts[i],x=v.x*w,y=v.y*h;
      const r=(.45+v.r)*.72,alpha=.11+.19*Math.sin(v.phase+2);
      disc(x,y,r,'rgba(96,158,214,'+alpha+')');
    }
    ctx.save();ctx.globalAlpha=.18;ctx.strokeStyle='#7aa1cc';
    for(let i=0;i<11;i++){
      ctx.beginPath();
      for(let j=0;j<55;j++){
        const x=j*w/54;
        const y=h*(.19+i*.061)+Math.sin(j*.20+i*.85+elapsed*.10)*18;
        if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }
    ctx.restore();
    // Hydrogen cloud glows subtly, leaving a genuinely darker epoch.
    nebula(w*.43,h*.59,R*.33,'64,80,126',.14);
  }
  function dawn(t){
    const R=Math.min(w,h);
    nebula(w*.56,h*.50,R*.72,'47,89,130',.29);
    nebula(w*.39,h*.39,R*.38,'146,60,93',.30);
    nebula(w*.76,h*.60,R*.42,'46,115,155',.25);
    const count=Math.floor(5+t*95);
    for(let i=0;i<count;i++){
      const p=pts[i*13%pts.length],g=nodes[i%nodes.length],a=p.phase+elapsed*.025;
      const dist=(.015+.13*p.x)*R*(1-.44*t);
      const x=g.x*w+Math.cos(a)*dist,y=g.y*h+Math.sin(a)*dist;
      star(x,y,.7+p.r*.75,.38+.3*t,i%4?'243,210,151':'125,207,255');
    }
    for(let k=0;k<Math.floor(3+10*t);k++){
      const g=nodes[(k*5+3)%nodes.length];
      star(g.x*w,g.y*h,1.6+2.4*t,.86,'255,214,159');
    }
  }
  function cosmicWeb(t){
    const R=Math.min(w,h),growth=.16+1.15*t;
    nebula(w*.47,h*.45,R*.78,'34,72,122',.34);
    nebula(w*.68,h*.63,R*.55,'48,113,160',.22);
    // First-order evolving Fourier displacement field.
    ctx.save();
    for(let i=0;i<pts.length;i++){
      const x=fract(qx[i]+growth*vx[i]*.16),y=fract(qy[i]+growth*vy[i]*.16);
      const r=pts[i].r*(.64+.3*t);
      disc(x*w,y*h,r,'rgba(143,214,240,'+(.18+.22*t)+')');
    }
    ctx.restore();
    // Visible filaments and knots are a schematic overlay for legibility.
    ctx.save();ctx.globalCompositeOperation='screen';
    edges.forEach((e,k)=>{
      const a=nodes[e.i],b=nodes[e.j];
      const x1=a.x*w,y1=a.y*h,x2=b.x*w,y2=b.y*h;
      const bend=Math.sin(e.phase+elapsed*.16)*R*.02;
      ctx.beginPath();ctx.moveTo(x1,y1);
      ctx.quadraticCurveTo((x1+x2)/2+bend,(y1+y2)/2-bend,x2,y2);
      ctx.strokeStyle='rgba(99,185,248,'+(.17+.19*t)+')';ctx.lineWidth=3.8+1.3*t;
      ctx.shadowBlur=12;ctx.shadowColor='rgba(69,142,215,.9)';ctx.stroke();
      ctx.shadowBlur=0;ctx.lineWidth=.9;
      ctx.strokeStyle='rgba(151,226,255,'+(.35+.26*t)+')';ctx.stroke();
    });
    ctx.restore();
    nodes.forEach((n,i)=>{
      const x=n.x*w,y=n.y*h;
      nebula(x,y,R*(.07+n.size*.008),'81,147,229',.17+.13*t);
      star(x,y,.8+n.size*.50,.57+.35*t,i%4===0?'255,216,159':'128,214,250');
      if(i%5===0){
        // An orbital galaxy motif progressively brightens at overdense nodes.
        ctx.save();ctx.translate(x,y);ctx.rotate(n.phase+elapsed*.065);
        for(let k=0;k<2;k++){
          ctx.beginPath();
          for(let j=0;j<72;j++){
            const ang=j*.098+k*Math.PI,r=(3+j*.25)*(.5+t*.7);
            const xx=Math.cos(ang)*r,yy=Math.sin(ang)*r*.55;
            if(!j)ctx.moveTo(xx,yy);else ctx.lineTo(xx,yy);
          }
          ctx.strokeStyle='rgba(203,225,255,'+(.19+.21*t)+')';
          ctx.lineWidth=1.45;ctx.stroke();
        }
        ctx.restore();
      }
    });
  }
  function renderScene(i,t){
    if(i===0)hot(t);
    else if(i===1)cmb(t);
    else if(i===2)darkAges(t);
    else if(i===3)dawn(t);
    else cosmicWeb(t);
  }
  function render(){
    base();
    const exact=cl(p,0,.999999)*5,idx=Math.min(4,Math.floor(exact));
    const t=exact-idx;
    renderScene(idx,t);
    // Smooth chapter transitions rather than an instantaneous hard cut.
    if(idx<4 && t>.77){
      ctx.save();ctx.globalAlpha=(t-.77)/.23;
      renderScene(idx+1,0);
      ctx.restore();
    }
    // A dark edge-to-edge frame ensures captions and star colours remain legible.
    const vignette=ctx.createRadialGradient(w*.54,h*.48,Math.min(w,h)*.13,w*.54,h*.48,Math.max(w,h)*.69);
    vignette.addColorStop(0,'rgba(0,0,0,0)');
    vignette.addColorStop(1,'rgba(1,5,16,.65)');
    ctx.fillStyle=vignette;ctx.fillRect(0,0,w,h);
    if(slider && document.activeElement!==slider)slider.value=String(Math.round(p*1000));
    if(redshift)redshift.textContent=labels[idx].z;
    if(epoch)epoch.textContent=labels[idx].name;
    if(stageNumber)stageNumber.textContent=String(idx+1).padStart(2,'0')+' / 05';
    if(stageName)stageName.textContent=labels[idx].name;
    if(stageSubtitle)stageSubtitle.textContent=labels[idx].sub;
    epochNodes.forEach((node,k)=>node.classList.toggle('current-epoch',k===idx));
  }
  function resize(){
    const box=canvas.getBoundingClientRect();
    w=Math.max(240,box.width);h=Math.max(200,box.height);
    dpr=Math.min(devicePixelRatio||1,1.5);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    render();
  }
  function tick(now){
    raf=0;
    if(!playing||document.hidden||!visible||reduced.matches)return;
    const dt=last?Math.min(.06,Math.max(0,(now-last)/1000)):0;
    elapsed+=dt;
    p=(p+dt/23)%1;
    last=now;
    render();
    raf=requestAnimationFrame(tick);
  }
  function sync(){
    if(raf)cancelAnimationFrame(raf);
    raf=0;last=0;
    if(play){play.textContent=playing?'Pause Ⅱ':'Play ▶';play.setAttribute('aria-label',playing?'Pause cosmic-history animation':'Play cosmic-history animation');}
    render();
    if(playing&&!document.hidden&&visible&&!reduced.matches)raf=requestAnimationFrame(tick);
  }
  if(play)play.addEventListener('click',()=>{playing=!playing;sync();});
  if(slider)slider.addEventListener('input',()=>{
    p=cl(Number(slider.value)/1000,0,.999999);
    playing=false;sync();
  });
  if('IntersectionObserver' in window){
    new IntersectionObserver(entries=>{
      visible=!!entries[0]&&entries[0].isIntersecting;sync();
    },{threshold:.01}).observe(canvas);
  }
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',sync);
  if(reduced.addEventListener)reduced.addEventListener('change',()=>{
    if(reduced.matches)playing=false;sync();
  });
  resize();sync();
})();

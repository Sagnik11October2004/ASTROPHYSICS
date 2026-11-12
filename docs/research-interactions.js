'use strict';
/* Interactive, explicitly illustrative density-profile laboratory and
   chapter buttons for the existing educational cosmic timeline.
   No production simulation data are loaded here. */
(() => {
  const svg=document.getElementById('dm-profile-plot');
  const slope=document.getElementById('dm-slope');
  const density=document.getElementById('dm-rho');
  const radius=document.getElementById('dm-radius');
  const readSlope=document.getElementById('dm-slope-value');
  const readRho=document.getElementById('dm-rho-value');
  const readRadius=document.getElementById('dm-radius-value');
  const readDensity=document.getElementById('dm-probe-density');
  const readEnhance=document.getElementById('dm-heating-ratio');
  const annotation=document.getElementById('dm-annotation');
  const reset=document.getElementById('dm-reset');
  const copy=document.getElementById('dm-copy');
  const copyStatus=document.getElementById('dm-copy-status');
  const NS='http://www.w3.org/2000/svg';
  const L=74,R=704,T=28,B=343;
  const minX=Math.log10(.05),maxX=1,minY=-1,maxY=8;
  const log10=Math.log10;
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const xMap=r=>L+(log10(r)-minX)/(maxX-minX)*(R-L);
  const yMap=d=>B-(log10(Math.max(1e-12,d))-minY)/(maxY-minY)*(B-T);
  const fmt=(v,n=2)=>Number(v).toLocaleString('en-US',{maximumSignificantDigits:n});
  function svgEl(tag,attrs,parent){
    const node=document.createElementNS(NS,tag);
    for(const [k,v] of Object.entries(attrs||{}))node.setAttribute(k,v);
    if(parent)parent.appendChild(node);
    return node;
  }
  let curves=null,probe=null;
  if(svg && slope && density && radius){
    const ticks=svgEl('g',{'class':'dm-plot-ticks'},svg);
    for(const r of [.05,.1,.3,1,3,10]){
      const x=xMap(r);
      svgEl('line',{x1:x,y1:T,x2:x,y2:B,'class':'dm-gridline'},ticks);
      const text=svgEl('text',{x,y:B+22,'text-anchor':'middle','class':'dm-axis-label'},ticks);
      text.textContent=String(r);
    }
    for(const pow of [0,2,4,6,8]){
      const y=yMap(Math.pow(10,pow));
      svgEl('line',{x1:L,y1:y,x2:R,y2:y,'class':'dm-gridline'},ticks);
      const text=svgEl('text',{x:L-13,y:y+4,'text-anchor':'end','class':'dm-axis-label'},ticks);
      text.textContent='10'+String.fromCharCode(0x2070+0); // replaced below by plain exponents
      text.textContent='1e'+pow;
    }
    svgEl('line',{x1:L,y1:B,x2:R,y2:B,'class':'dm-axis'},ticks);
    svgEl('line',{x1:L,y1:T,x2:L,y2:B,'class':'dm-axis'},ticks);
    const xlabel=svgEl('text',{x:(L+R)/2,y:B+52,'text-anchor':'middle','class':'dm-axis-title'},ticks);
    xlabel.textContent='Radius r [pc] · logarithmic';
    const ylabel=svgEl('text',{x:22,y:(T+B)/2,transform:'rotate(-90 22 '+((T+B)/2)+')','text-anchor':'middle','class':'dm-axis-title'},ticks);
    ylabel.textContent='Dark-matter density [GeV cm⁻³]';
    curves=svgEl('g',{},svg);
    svgEl('path',{id:'dm-reference-curve','class':'dm-reference-curve'},curves);
    svgEl('path',{id:'dm-model-curve','class':'dm-model-curve'},curves);
    probe=svgEl('g',{'class':'dm-plot-probe'},svg);
    svgEl('line',{id:'dm-probe-line',y1:T,y2:B,'class':'dm-probe-line'},probe);
    svgEl('circle',{id:'dm-probe-dot',r:6,'class':'dm-probe-dot'},probe);
    const guide=svgEl('rect',{x:L,y:T,width:R-L,height:B-T,fill:'transparent','class':'dm-plot-hit',role:'presentation'},svg);
    const setProbe=e=>{
      const box=svg.getBoundingClientRect();
      if(!box.width)return;
      const px=(e.clientX-box.left)/box.width*760;
      const logR=minX+clamp((px-L)/(R-L),0,1)*(maxX-minX);
      radius.value=String(logR);draw();
    };
    guide.addEventListener('pointerdown',setProbe);
    guide.addEventListener('pointermove',e=>{if(e.buttons===1)setProbe(e);});
    [slope,density,radius].forEach(input=>input.addEventListener('input',draw));
    if(reset)reset.addEventListener('click',()=>{
      slope.value='1.82'; density.value=String(log10(1770));radius.value='-1';draw();
      if(copyStatus)copyStatus.textContent='';
    });
    if(copy)copy.addEventListener('click',async()=>{
      const g=Number(slope.value),rho=Math.pow(10,Number(density.value)),r=Math.pow(10,Number(radius.value));
      const s='Illustrative DM cusp: rho(r)='+rho.toPrecision(5)+' (r/1 pc)^(-'+g.toFixed(2)+') GeV cm^-3; probe r='+r.toPrecision(3)+' pc. Not a validated heating or stellar evolution prediction.';
      try{
        await navigator.clipboard.writeText(s);
        if(copyStatus)copyStatus.textContent='Parameters copied.';
      }catch(_){if(copyStatus)copyStatus.textContent='Copy unavailable in this browser.';}
    });
    draw();
  }
  function modelPath(rho1,gamma){
    const pts=[];
    for(let i=0;i<=180;i++){
      const logR=minX+(maxX-minX)*i/180;
      const r=Math.pow(10,logR),d=rho1*Math.pow(r,-gamma);
      pts.push((i?'L':'M')+xMap(r).toFixed(2)+','+clamp(yMap(d),T,B).toFixed(2));
    }
    return pts.join(' ');
  }
  function draw(){
    if(!svg||!curves||!probe)return;
    const gamma=Number(slope.value);
    const rho1=Math.pow(10,Number(density.value));
    const r=Math.pow(10,Number(radius.value));
    const dm=rho1*Math.pow(r,-gamma);
    const ref=rho1/r;
    const ratio=Math.pow(dm/ref,2);
    const best=curves.querySelector('#dm-model-curve');
    const baseline=curves.querySelector('#dm-reference-curve');
    if(best)best.setAttribute('d',modelPath(rho1,gamma));
    if(baseline)baseline.setAttribute('d',modelPath(rho1,1));
    const line=probe.querySelector('#dm-probe-line'),dot=probe.querySelector('#dm-probe-dot');
    if(line){line.setAttribute('x1',xMap(r));line.setAttribute('x2',xMap(r));}
    if(dot){dot.setAttribute('cx',xMap(r));dot.setAttribute('cy',clamp(yMap(dm),T,B));}
    if(readSlope)readSlope.textContent=gamma.toFixed(2);
    if(readRho)readRho.textContent=fmt(rho1,4);
    if(readRadius)readRadius.textContent=r.toFixed(r<1?3:2);
    if(readDensity)readDensity.textContent=fmt(dm,4);
    if(readEnhance)readEnhance.textContent=fmt(ratio,3)+'×';
    if(annotation)annotation.textContent='At '+r.toPrecision(3)+' pc, annihilation power per unit volume would scale as ρ²: '+fmt(ratio,3)+'× the r⁻¹ reference, holding other particle-physics factors fixed.';
  }

  const chapters=[
    'CMB: photon decoupling near z ≈ 1100, well before the TNG300 film begins.',
    'Initial conditions: nearly homogeneous primordial matter at z = 499; this is an educational stage, not TNG300 footage.',
    'Minihaloes: small dark-matter haloes assemble and primordial gas cools and collapses.',
    'First stars: gas collapses into metal-free protostars; annihilation-supported growth is a conditional hypothesis.',
    'Galaxies: hierarchical assembly and ionizing sources transform the intergalactic medium.',
    'Today: the cosmic web connects galaxies, groups and clusters at z ≈ 0.'
  ];
  const positions=[25,140,315,510,705,905];
  const description=document.getElementById('cosmos-chapter-description');
  const epochButtons=[...document.querySelectorAll('button.epoch[data-epoch-index]')];
  const preview=document.getElementById('cosmo-toggle-model');
  const slider=document.getElementById('cosmo-progress');
  epochButtons.forEach((button,index)=>button.addEventListener('click',()=>{
    epochButtons.forEach((el,i)=>el.setAttribute('aria-pressed',String(i===index)));
    if(description)description.textContent=chapters[index]+' Select another chapter to explore the illustrated timeline.';
    if(preview && !document.body.classList.contains('cosmo-model-active'))preview.click();
    const ready=window.researchLoad?window.researchLoad('./cosmic-story.js?v=6'):Promise.resolve();
    ready.then(()=>{
      if(slider){
        slider.value=positions[index];
        slider.dispatchEvent(new Event('input',{bubbles:true}));
      }
    }).catch(()=>{if(description)description.textContent='The educational timeline could not be loaded. '+chapters[index];});
  }));
})();

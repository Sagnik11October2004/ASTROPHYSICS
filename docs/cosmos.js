'use strict';

/* Restored personal-site interactions and an original educational cosmic-time
   illustration. Entirely client-side: safe for static GitHub Pages hosting. */
(function () {
  // Restore the previous portfolio's filterable research cards.
  const filterButtons = Array.from(document.querySelectorAll('.research-filter'));
  const projects = Array.from(document.querySelectorAll('.work-card'));
  const empty = document.getElementById('research-empty');
  function filterResearch(value) {
    let count = 0;
    projects.forEach(function (card) {
      const tags = (card.dataset.topic || '').split(/\s+/);
      const show = value === 'all' || tags.includes(value);
      card.hidden = !show;
      if (show) count += 1;
    });
    filterButtons.forEach(function (button) {
      const active = button.dataset.filter === value;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    if (empty) empty.hidden = count !== 0;
  }
  filterButtons.forEach(function (button) {
    button.addEventListener('click', function () { filterResearch(button.dataset.filter); });
  });

  const printButton = document.getElementById('print-profile');
  if (printButton) printButton.addEventListener('click', function () { window.print(); });

  // Country-level markers used in the original "Travel With Me" section.
  // Coordinates are *map locators* for countries, not claims about visited cities.
  const countries = [
    { name: 'India', lat: 22.6, lon: 78.9, region: 'india', detail: 'Kolkata · Nainital · Kanpur' },
    { name: 'Sweden', lat: 59.7, lon: 15.0, region: 'northern', detail: 'Gothenburg · Stockholm · Summer research 2026' },
    { name: 'Denmark', lat: 56.0, lon: 10.0, region: 'northern' },
    { name: 'Finland', lat: 64.5, lon: 26.0, region: 'northern' },
    { name: 'Netherlands', lat: 52.2, lon: 5.4, region: 'northern' },
    { name: 'Norway', lat: 62.0, lon: 9.0, region: 'northern' }
  ];
  const regionButtons = Array.from(document.querySelectorAll('.travel-filter'));
  const countryButtons = Array.from(document.querySelectorAll('#country-list button'));
  const mapFallback = document.getElementById('map-fallback');
  const travelContainer = document.getElementById('travel-map');
  let travelMap = null;
  let mapMarkers = new Map();
  let region = 'all';

  function visibleCountries() { return countries.filter(function (item) { return region === 'all' || item.region === region; }); }
  function selectCountry(countryName) {
    countryButtons.forEach(function (button) {
      button.classList.toggle('active', button.dataset.country === countryName);
    });
    const item = countries.find(function (place) { return place.name === countryName; });
    if (!item || !travelMap) return;
    const marker = mapMarkers.get(item.name);
    if (!marker) return;
    travelMap.flyTo([item.lat,item.lon], region === 'india' ? 4 : 5, { duration: .75 });
    marker.openPopup();
  }
  function setRegion(value) {
    region = value;
    regionButtons.forEach(function (button) {
      const active = button.dataset.region === value;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    countryButtons.forEach(function (button) {
      const item = countries.find(function (place) { return place.name === button.dataset.country; });
      button.hidden = !!item && value !== 'all' && item.region !== value;
      button.classList.remove('active');
    });
    if (!travelMap || !window.L) return;
    mapMarkers.forEach(function (marker, countryName) {
      const place = countries.find(function (entry) { return entry.name === countryName; });
      if (!place) return;
      if (value === 'all' || place.region === value) marker.addTo(travelMap);
      else travelMap.removeLayer(marker);
    });
    const visible = visibleCountries();
    if (value === 'all') {
      travelMap.setView([19,12], 2, { animate: true });
    } else if (visible.length) {
      const bounds = window.L.latLngBounds(visible.map(function (item) { return [item.lat,item.lon]; }));
      travelMap.fitBounds(bounds.pad(.45),{ maxZoom: 5,animate: true });
    }
  }
  regionButtons.forEach(function (button) {
    button.addEventListener('click', function () { setRegion(button.dataset.region); });
  });
  countryButtons.forEach(function (button) {
    button.addEventListener('click', function () { selectCountry(button.dataset.country); });
  });

  if (travelContainer && window.L) {
    try {
      const L = window.L;
      travelMap = L.map(travelContainer, {
        zoomControl: true, scrollWheelZoom: true, tap: false, worldCopyJump: true, minZoom: 2, maxZoom: 12
      });
      // CARTO's dark OSM-derived basemap. Attribution remains visible.
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a> &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener noreferrer">CARTO</a>',
        subdomains: 'abcd', maxZoom: 19
      }).addTo(travelMap);
      countries.forEach(function (item) {
        const icon = L.divIcon({ className: '', html: '<div class="travel-marker"></div>', iconSize: [20,20], iconAnchor: [10,10] });
        const marker = L.marker([item.lat,item.lon], { icon: icon, title:item.name })
          .bindPopup('<strong>' + item.name + '</strong><p>' + (item.detail || 'Country on my travel map') + '</p>', { className: 'travel-popup' })
          .on('click',function () { selectCountry(item.name); });
        marker.addTo(travelMap);
        mapMarkers.set(item.name,marker);
      });
      setRegion('all');
    } catch (error) {
      if (mapFallback) mapFallback.hidden = false;
    }
  } else if (mapFallback) {
    mapFallback.hidden = false;
  }
  const worldReset = document.getElementById('world-reset');
  if (worldReset) worldReset.addEventListener('click', function () {
    setRegion('all');
    if (travelMap) travelMap.invalidateSize();
  });


  /* LIVE PHYSICS EXPERIMENT — linear Zel'dovich-approximation 2D structure
     formation from a deterministic Gaussian-like displacement field.
     This evolves continuous trajectories x(q,a) = q + D(a) psi(q).
     It is not a full three-dimensional N-body, hydro or GR simulation.
     Pre-recombination visual effects and discrete galaxy lights are illustrative.
     All computation is local, with no external assets or backend. */
  const canvas = document.getElementById('cosmos-canvas');
  if (!canvas || !canvas.getContext) return;
  const ctx=canvas.getContext('2d',{alpha:false});
  if (!ctx) return;
  const scr=document.getElementById('cosmo-progress');
  const btn=document.getElementById('cosmo-play');
  const redshiftOutput=document.getElementById('cosmo-redshift');
  const epochOutput=document.getElementById('cosmo-epoch-readout');
  const epochNodes=Array.from(document.querySelectorAll('.epoch'));
  const motionQuery=window.matchMedia('(prefers-reduced-motion: reduce)');
  let w=800,h=650,dpr=1,visible=true,runId=0,lastTime=0,playing=!motionQuery.matches,p=0.025,accumulated=0;
  let scrW=300,scrH=175,bufferCanvas=document.createElement('canvas'),bufferCtx=bufferCanvas.getContext('2d',{willReadFrequently:true});
  if (!bufferCtx) return;
  const NX=104, NY=65, NUM=NX*NY;
  const qx=new Float32Array(NUM),qy=new Float32Array(NUM),vx=new Float32Array(NUM),vy=new Float32Array(NUM);
  const density=new Float32Array(300*190);
  let seed=201126;
  function rnd(){ seed=(1664525*seed+1013904223)>>>0;return seed/4294967296; }
  const modes=[];
  for(let j=0;j<48;j++){
    let kx=Math.floor(rnd()*13)-6, ky=Math.floor(rnd()*11)-5;
    if(kx===0&&ky===0)kx=1;
    const k=Math.sqrt(kx*kx+ky*ky);
    modes.push({kx:kx,ky:ky,phase:rnd()*Math.PI*2,amp:(.5+rnd())/Math.pow(k,2.65)});
  }
  let maxV=0;
  for(let j=0;j<NY;j++) for(let i=0;i<NX;i++){
    const n=j*NX+i, x=(i+.5)/NX,y=(j+.5)/NY;
    let u=0,v=0;
    for(const m of modes){
      const ph=2*Math.PI*(m.kx*x+m.ky*y)+m.phase;
      const a=m.amp*Math.sin(ph);
      u+=m.kx*a;v+=m.ky*a;
    }
    qx[n]=x;qy[n]=y;vx[n]=u;vy[n]=v;
    maxV=Math.max(maxV,Math.hypot(u,v));
  }
  for(let i=0;i<NUM;i++){vx[i]/=maxV||1;vy[i]/=maxV||1;}
  function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
  function epochAt(t){
    if(t<.16)return {i:0,name:'Hot early universe',z:'z ≫ 1100',detail:'Primordial plasma',g:0};
    if(t<.30)return {i:1,name:'Recombination / CMB',z:'z ≈ 1100',detail:'Recombination (~380 kyr)',g:0.02};
    if(t<.43)return {i:2,name:'Cosmic dark ages',z:'1100 → 30',detail:'Neutral gas',g:.04+(t-.30)*.24};
    const u=(t-.43)/.57;
    const a=1/31 + (1-1/31)*Math.pow(u,2.2);
    const z=1/a-1;
    return {i:u<.24?3:4,name:u<.24?'Cosmic dawn':'Cosmic-web growth',z:'z ≈ '+z.toFixed(z>5?1:2),detail:'Growing density perturbations',g:.13+Math.pow(u,.9)*1.74};
  }
  function resize(){
    const r=canvas.getBoundingClientRect();w=Math.max(300,r.width);h=Math.max(350,r.height);
    dpr=Math.min(devicePixelRatio||1,1.5);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    scrW=Math.min(365,Math.max(200,Math.round(w*.44)));
    scrH=Math.min(244,Math.max(142,Math.round(scrW*h/w)));
    bufferCanvas.width=scrW;bufferCanvas.height=scrH;
    if(!playing)draw();
  }
  function draw(){
    const data=epochAt(p);
    const background=ctx.createLinearGradient(0,0,w,h);
    background.addColorStop(0,'#040916');background.addColorStop(.55,'#070e1a');background.addColorStop(1,'#01050c');
    ctx.fillStyle=background;ctx.fillRect(0,0,w,h);
    ctx.save();
    const cx=w*.77,cy=h*.43,R=Math.min(w,h)*.72;
    const glow=ctx.createRadialGradient(cx,cy,0,cx,cy,R);
    if(data.i===0){glow.addColorStop(0,'rgba(243,171,86,.48)');glow.addColorStop(.5,'rgba(151,71,54,.2)');}
    else if(data.i===1){glow.addColorStop(0,'rgba(246,208,150,.18)');glow.addColorStop(.5,'rgba(63,90,156,.17)');}
    else {glow.addColorStop(0,'rgba(44,89,144,.28)');glow.addColorStop(.55,'rgba(15,52,100,.15)');}
    glow.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=glow;ctx.fillRect(0,0,w,h);
    // Periodically remap Lagrangian particles with linear growing-mode displacements.
    density.fill(0);
    const strength=data.g*.22;
    for(let n=0;n<NUM;n++){
      const x=((qx[n]+strength*vx[n])%1+1)%1;
      const y=((qy[n]+strength*vy[n])%1+1)%1;
      const ix=Math.min(scrW-2,Math.floor(x*scrW)),iy=Math.min(scrH-2,Math.floor(y*scrH));
      const k=iy*scrW+ix;
      density[k]+=1; density[k+1]+=.46;
      if(iy+1<scrH){density[k+scrW]+=.46;density[k+scrW+1]+=.23;}
    }
    const pixels=bufferCtx.createImageData(scrW,scrH);
    let peak=0;for(let n=0;n<scrW*scrH;n++)peak=Math.max(peak,density[n]);
    const inv=1/Math.max(2.5,peak*.8);
    for(let i=0;i<scrW*scrH;i++){
      const d=Math.pow(clamp(density[i]*inv,0,1),.58);
      const j=i*4;
      const age=clamp((p-.23)/.77,0,1);
      const luminous=d*(.12+age*.88);
      pixels.data[j]=Math.floor((data.i<2?249:95)*luminous+6);
      pixels.data[j+1]=Math.floor((data.i<2?169:196)*luminous+10);
      pixels.data[j+2]=Math.floor((data.i<2?105:252)*luminous+19);
      pixels.data[j+3]=255;
    }
    bufferCtx.putImageData(pixels,0,0);
    ctx.globalAlpha=data.i===0?.6:data.i===1?.55:data.i===2?.45:.9;
    ctx.imageSmoothingEnabled=true;
    ctx.drawImage(bufferCanvas,0,0,w,h);
    ctx.globalAlpha=1;
    // Recombination / plasma acoustic fluctuations shown schematically.
    if(data.i<=1) {
      ctx.globalCompositeOperation='screen';
      for(let k=0;k<11;k++){
        const x=(k/11+.05)*w;
        ctx.beginPath();ctx.strokeStyle='rgba(255,185,115,'+(data.i===0?.12:.045)+')';
        ctx.lineWidth=1.5;
        for(let y=0;y<h;y+=9){
          const xx=x+Math.sin(y*.015+k*1.8+accumulated)*23;
          if(y===0)ctx.moveTo(xx,y);else ctx.lineTo(xx,y);
        }
        ctx.stroke();
      }
      ctx.globalCompositeOperation='source-over';
    }
    // Density peaks may host representative luminous objects after cosmic dawn.
    if(data.i>=3) {
      const stars=Math.floor(clamp((p-.43)/.57,0,1)*135);
      for(let k=0;k<stars;k++){
        const n=(k*1171+53)%NUM;
        const xx=(((qx[n]+strength*vx[n])%1+1)%1)*w;
        const yy=(((qy[n]+strength*vy[n])%1+1)%1)*h;
        if((k*13)%7>3)continue;
        const r=(k%5===0?2.25:1.1)*(1+.15*Math.sin(accumulated*2+k));
        ctx.beginPath();ctx.fillStyle='rgba(255,220,156,.85)';
        ctx.shadowColor='rgba(255,178,100,.9)';ctx.shadowBlur=12;
        ctx.arc(xx,yy,r,0,Math.PI*2);ctx.fill();
      }
      ctx.shadowBlur=0;
    }
    // Cosmic background points in comoving visual coordinates.
    for(let j=0;j<160;j++){
      const x=((j*173.713)%997)/997*w,y=((j*317.471)%991)/991*h;
      const opacity=data.i<2?.15:.23+(j%7)/60;
      ctx.fillStyle='rgba(217,231,247,'+opacity+')';
      ctx.fillRect(x,y,(j%11===0?1.8:.8),(j%11===0?1.8:.8));
    }
    ctx.restore();
    // Nonphysical abstract epoch boundary is never used as time-scaled data.
    if(redshiftOutput)redshiftOutput.textContent=data.z;
    if(epochOutput)epochOutput.textContent=data.name;
    epochNodes.forEach(function(node,i){node.classList.toggle('current-epoch',i===data.i);});
    if(scr && document.activeElement!==scr)scr.value=Math.round(p*1000);
  }
  function frame(now){
    runId=0;
    if(!playing||document.hidden||!visible||motionQuery.matches)return;
    const delta=lastTime?Math.min(.1,(now-lastTime)/1000):0;
    if(delta >=0){
      p=(p+delta/33)%1;
      accumulated+=delta;
    }
    lastTime=now;draw();
    runId=requestAnimationFrame(frame);
  }
  function updatePlayButton(){
    if(btn){btn.textContent=playing?'Pause Ⅱ':'Play ▶';btn.setAttribute('aria-label',playing?'Pause cosmological simulation':'Play cosmological simulation');}
  }
  function sync(){
    if(runId)cancelAnimationFrame(runId);
    runId=0;lastTime=0;
    updatePlayButton();draw();
    if(playing&&!document.hidden&&visible&&!motionQuery.matches)runId=requestAnimationFrame(frame);
  }
  if(btn)btn.addEventListener('click',function(){playing=!playing;sync();});
  if(scr)scr.addEventListener('input',function(){p=clamp(Number(scr.value)/1000,0,1);playing=false;sync();});
  if('IntersectionObserver' in window){
    new IntersectionObserver(function(entries){visible=!!entries[0]&&entries[0].isIntersecting;sync();},{threshold:.01}).observe(canvas);
  }
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',sync);
  if(motionQuery.addEventListener)motionQuery.addEventListener('change',function(){if(motionQuery.matches)playing=false;sync();});
  resize();sync();
})();

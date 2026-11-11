'use strict';

/* Research filtering and a self-contained, draggable offline world atlas.
   All map geography is bundled as world.svg. No Leaflet, tiles, API keys or CDN. */
(function () {
  const researchButtons = Array.from(document.querySelectorAll('.research-filter'));
  const researchCards = Array.from(document.querySelectorAll('.work-card'));
  const empty = document.getElementById('research-empty');
  function filterResearch(topic) {
    let found=0;
    researchCards.forEach(function(card){
      const visible=topic==='all'||(card.dataset.topic||'').split(/\s+/).includes(topic);
      card.hidden=!visible;if(visible)found++;
    });
    researchButtons.forEach(function(button){
      const active=button.dataset.filter===topic;
      button.classList.toggle('active',active);
      button.setAttribute('aria-pressed',String(active));
    });
    if(empty)empty.hidden=found!==0;
  }
  researchButtons.forEach(function(button){
    button.addEventListener('click',function(){filterResearch(button.dataset.filter);});
  });
  const print=document.getElementById('print-profile');
  if(print)print.addEventListener('click',function(){window.print();});

  const map=document.getElementById('travel-map');
  const world=document.getElementById('atlas-stage');
  const details=document.getElementById('map-details');
  if(!map||!world)return;
  const pins=Array.from(map.querySelectorAll('.atlas-pin'));
  const countryButtons=Array.from(document.querySelectorAll('#country-list button'));
  const regionButtons=Array.from(document.querySelectorAll('.travel-filter'));
  const locations={
    India:{lon:78.9,lat:22.6,region:'india',info:'India · IISER Kolkata, IIT Kanpur and ARIES Nainital'},
    Sweden:{lon:15.0,lat:59.7,region:'northern',info:'Sweden · Chalmers (Gothenburg) and Nordita (Stockholm)'},
    Denmark:{lon:10.0,lat:56.0,region:'northern',info:'Denmark · Northern Europe'},
    Finland:{lon:26.0,lat:64.5,region:'northern',info:'Finland · Northern Europe'},
    Netherlands:{lon:5.4,lat:52.2,region:'northern',info:'Netherlands · Northern Europe'},
    Norway:{lon:9.0,lat:62.0,region:'northern',info:'Norway · Northern Europe'}
  };
  let scale=1,tx=0,ty=0,region='all',active=null;
  function clamp(v,min,max){return Math.max(min,Math.min(max,v));}
  function constrain(){
    const w=map.clientWidth,h=map.clientHeight;
    tx=clamp(tx,w*(1-scale),0);ty=clamp(ty,h*(1-scale),0);
    if(scale===1){tx=0;ty=0;}
  }
  function draw(transformNow){
    constrain();
    world.style.transition=transformNow?'transform .42s ease-out':'none';
    world.style.transform='translate('+tx.toFixed(1)+'px,'+ty.toFixed(1)+'px) scale('+scale.toFixed(3)+')';
    pins.forEach(function(pin){pin.style.setProperty('--pin-scale',(1/scale).toFixed(4));});
  }
  function zoomTo(next,x,y){
    const w=map.clientWidth,h=map.clientHeight;
    const focusX=x===undefined?w/2:x,focusY=y===undefined?h/2:y;
    const old=scale;scale=clamp(next,1,7);
    tx=focusX-(focusX-tx)*scale/old;
    ty=focusY-(focusY-ty)*scale/old;
    draw(true);
  }
  function pick(name,move){
    const loc=locations[name];
    if(!loc)return;
    active=name;
    pins.forEach(function(pin){
      const selected=pin.dataset.country===name;
      pin.classList.toggle('active',selected);
      pin.setAttribute('aria-pressed',String(selected));
    });
    countryButtons.forEach(function(button){button.classList.toggle('active',button.dataset.country===name);});
    if(details)details.textContent=loc.info+' · Select another marker, zoom or drag to explore.';
    if(move){
      const x=(loc.lon+180)/360*map.clientWidth;
      const y=(85-loc.lat)/170*map.clientHeight;
      scale=name==='India'?2.5:Math.max(3.0,scale);
      tx=map.clientWidth*.5-scale*x;
      ty=map.clientHeight*.52-scale*y;
      draw(true);
    }
  }
  function reset(){
    scale=1;tx=0;ty=0;active=null;region='all';
    pins.forEach(function(pin){pin.hidden=false;pin.classList.remove('active');pin.setAttribute('aria-pressed','false');});
    countryButtons.forEach(function(button){button.hidden=false;button.classList.remove('active');});
    regionButtons.forEach(function(button){
      const active=button.dataset.region==='all';
      button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));
    });
    if(details)details.textContent='Full-world view · Vector geographic map. Choose a marked country or zoom to investigate.';
    draw(true);
  }
  function selectRegion(value){
    if(value==='all'){reset();return;}
    region=value;
    regionButtons.forEach(function(button){
      const selected=button.dataset.region===value;
      button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));
    });
    const locationList=Object.entries(locations).filter(function(entry){return entry[1].region===value;});
    pins.forEach(function(pin){pin.hidden=locations[pin.dataset.country]?.region!==value;});
    countryButtons.forEach(function(button){button.hidden=locations[button.dataset.country]?.region!==value;});
    if(locationList.length===1){pick(locationList[0][0],true);}
    else{
      active=null;
      pins.forEach(function(pin){pin.classList.remove('active');});
      countryButtons.forEach(function(button){button.classList.remove('active');});
      scale=4.35;
      const x=(17.0+180)/360*map.clientWidth,y=(85-59)/170*map.clientHeight;
      tx=map.clientWidth*.5-scale*x;
      ty=map.clientHeight*.5-scale*y;
      draw(true);
      if(details)details.textContent='Northern Europe · Select a country marker to view its details.';
    }
  }
  pins.forEach(function(pin){
    pin.setAttribute('aria-pressed','false');
    pin.addEventListener('click',function(e){e.stopPropagation();pick(pin.dataset.country,true);});
  });
  countryButtons.forEach(function(button){
    button.addEventListener('click',function(){pick(button.dataset.country,true);});
  });
  regionButtons.forEach(function(button){
    button.addEventListener('click',function(){selectRegion(button.dataset.region);});
  });
  const worldReset=document.getElementById('world-reset');
  if(worldReset)worldReset.addEventListener('click',reset);
  const plus=document.getElementById('atlas-zoom-in');
  const minus=document.getElementById('atlas-zoom-out');
  if(plus)plus.addEventListener('click',function(){zoomTo(scale*1.5);});
  if(minus)minus.addEventListener('click',function(){zoomTo(scale/1.5);});
  map.addEventListener('wheel',function(e){
    if(e.ctrlKey||Math.abs(e.deltaY)>0){
      e.preventDefault();
      const box=map.getBoundingClientRect();
      zoomTo(scale*Math.exp(-e.deltaY*.001),e.clientX-box.left,e.clientY-box.top);
    }
  },{passive:false});
  let drag=null;
  map.addEventListener('pointerdown',function(e){
    if(e.target.closest('button'))return;
    drag={x:e.clientX,y:e.clientY,tx:tx,ty:ty};
    map.classList.add('is-dragging');world.classList.add('is-grabbing');
    if(map.setPointerCapture)map.setPointerCapture(e.pointerId);
  });
  map.addEventListener('pointermove',function(e){
    if(!drag)return;
    tx=drag.tx+(e.clientX-drag.x);
    ty=drag.ty+(e.clientY-drag.y);
    draw(false);
  });
  function stopDrag(){
    drag=null;map.classList.remove('is-dragging');world.classList.remove('is-grabbing');
  }
  map.addEventListener('pointerup',stopDrag);
  map.addEventListener('pointercancel',stopDrag);
  map.addEventListener('lostpointercapture',stopDrag);
  window.addEventListener('resize',function(){
    // Preserve geographic centering as the viewport changes.
    if(active)pick(active,true);else if(region==='northern')selectRegion(region);else draw(false);
  },{passive:true});
  const svg=map.querySelector('.atlas-world');
  if(svg)svg.addEventListener('error',function(){
    if(details)details.textContent='The bundled coastline image could not be loaded. The country buttons remain available.';
  });
  reset();
})();

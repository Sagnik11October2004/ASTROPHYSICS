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
    { name: 'India', lat: 22.6, lon: 78.9, region: 'india' },
    { name: 'Sweden', lat: 62.0, lon: 15.0, region: 'northern' },
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
    if (visible.length) {
      const bounds = window.L.latLngBounds(visible.map(function (item) { return [item.lat,item.lon]; }));
      travelMap.fitBounds(bounds.pad(.48),{ maxZoom: 4,animate: true });
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
        zoomControl: true, scrollWheelZoom: false, tap: false, worldCopyJump: true
      });
      // CARTO's dark OSM-derived basemap. Attribution remains visible.
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a> &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener noreferrer">CARTO</a>',
        subdomains: 'abcd', maxZoom: 19
      }).addTo(travelMap);
      countries.forEach(function (item) {
        const icon = L.divIcon({ className: '', html: '<div class="travel-marker"></div>', iconSize: [20,20], iconAnchor: [10,10] });
        const marker = L.marker([item.lat,item.lon], { icon: icon, title:item.name })
          .bindPopup('<strong>' + item.name + '</strong>', { className: 'travel-popup' })
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

  // The cosmic backdrop depicts five evolutionary eras spatially.
  // It is original illustrative artwork, not an N-body or radiative-transfer run.
  const canvas = document.getElementById('cosmos-canvas');
  const epochNodes = Array.from(document.querySelectorAll('.epoch'));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d',{alpha:false});
  if (!ctx) return;

  let width = 1000, height = 700, visible = true, frame = 0, frameRequest = 0;
  let lastEpoch = -1;
  let lastTime = 0;
  let canvasScale = 1;

  function seeded(seed) {
    let state = seed >>> 0;
    return function () {
      state = (Math.imul(state,1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
  }
  const random = seeded(137913);
  const backgroundStars = Array.from({length:240},function () {
    return { x:random(), y:random(), size:random()*1.3+.25, a:random(), phase:random()*Math.PI*2 };
  });
  const gas = Array.from({length:150},function () {
    return {x:random(),y:random(),size:.3+random()*2.7,phase:random()*6.28,speed:.2+random()*.8};
  });
  const webCenters = [
    [.63,.19],[.75,.48],[.90,.22],[.92,.75],[.66,.81],[.84,.62],[.98,.45]
  ];
  const cosmicNodes = Array.from({length:101},function (_,index) {
    const c = webCenters[index % webCenters.length];
    const a = random()*Math.PI*2, r = Math.sqrt(random())*.15;
    return {
      x:Math.min(1.06,Math.max(.52,c[0]+Math.cos(a)*r)),
      y:Math.max(-.08,Math.min(1.1,c[1]+Math.sin(a)*r)),
      r: random()*1.8+.6, t:random()*6.283,
      a:.2+random()*.6
    };
  });
  const edges = [];
  for (let i = 0; i < cosmicNodes.length; i++) {
    for (let j = i+1; j < cosmicNodes.length; j++) {
      const dx = cosmicNodes[i].x - cosmicNodes[j].x;
      const dy = cosmicNodes[i].y - cosmicNodes[j].y;
      const d = Math.sqrt(dx*dx+dy*dy);
      if (d < .082 && edges.length < 410) edges.push({i:i,j:j,alpha:.08+(.082-d)*2.9});
    }
  }
  function glow(x,y,rgb,radius,alpha) {
    const g = ctx.createRadialGradient(x,y,0,x,y,radius);
    g.addColorStop(0,'rgba('+rgb+','+alpha+')');
    g.addColorStop(.36,'rgba('+rgb+','+(alpha*.4)+')');
    g.addColorStop(1,'rgba('+rgb+',0)');
    ctx.fillStyle = g;
    ctx.fillRect(x-radius,y-radius,radius*2,radius*2);
  }
  function colorFill(x,y,radius,color,a) {
    ctx.beginPath();
    ctx.arc(x,y,radius,0,Math.PI*2);
    ctx.fillStyle='rgba('+color+','+a+')';
    ctx.fill();
  }
  function drawBackground(t) {
    const W=width,H=height;
    ctx.fillStyle='#040a13';
    ctx.fillRect(0,0,W,H);
    const deep = ctx.createLinearGradient(0,H*.2,W,H*.7);
    deep.addColorStop(0,'rgba(10,25,50,.38)');
    deep.addColorStop(.5,'rgba(12,35,60,.22)');
    deep.addColorStop(1,'rgba(10,40,56,.7)');
    ctx.fillStyle=deep;ctx.fillRect(0,0,W,H);

    // A primordial, expanding hot plasma cone.
    glow(W*.10,H*.43,'255,164,96',Math.max(W*.21,190),.34);
    glow(W*.20,H*.54,'232,111,74',W*.18,.18);
    glow(W*.30,H*.40,'133,154,215',W*.17,.12);
    ctx.save();
    ctx.globalCompositeOperation='screen';
    for(let k=0;k<19;k++){
      const radius=H*(.025 + k*.027);
      ctx.beginPath();
      ctx.ellipse(W*.12,H*.44,radius*.61,radius,-.15,Math.PI*.44,Math.PI*1.71);
      ctx.strokeStyle='rgba(255,186,126,'+Math.max(.009,.14-k*.006)+')';
      ctx.lineWidth=.7+(k%4===0?1.2:0);
      ctx.stroke();
    }
    ctx.restore();

    // Recombination: a gently mottled last-scattering-surface motif.
    const cx=W*.34,cy=H*.50,r=Math.min(W*.125,H*.205);
    glow(cx,cy,'147,211,225',r*1.9,.13);
    ctx.save();
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.clip();
    const cmb=ctx.createRadialGradient(cx-r*.4,cy-r*.5,r*.1,cx,cy,r*1.2);
    cmb.addColorStop(0,'rgba(156,204,220,.22)');
    cmb.addColorStop(.5,'rgba(76,139,183,.16)');
    cmb.addColorStop(1,'rgba(30,75,117,.05)');
    ctx.fillStyle=cmb;ctx.fillRect(cx-r,cy-r,2*r,2*r);
    for(let i=0;i<110;i++) {
      const a=i*2.39996, radius=r*Math.sqrt((i+.5)/110);
      colorFill(cx+radius*Math.cos(a),cy+radius*Math.sin(a),r*(.045+(i%7)/84),i%4===0?'250,178,125':'133,207,237',.03+(i%5)*.016);
    }
    ctx.restore();
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle='rgba(174,213,235,.24)';ctx.lineWidth=1;ctx.stroke();

    // Dark ages are represented by sparse, dim hydrogen-cloud filaments.
    for(let i=0;i<gas.length;i++){
      const p=gas[i],x=W*(.39+.19*p.x),y=H*p.y;
      const shimmer=.7+.3*Math.sin(t*p.speed+p.phase);
      colorFill(x,y,p.size*.68,'115,173,197',.085*shimmer);
    }

    // Cosmic dawn / reionization: isolated first stellar regions.
    const dawn = [[.56,.26],[.61,.54],[.68,.70],[.74,.33],[.78,.81],[.68,.15]];
    dawn.forEach(function (p,index){
      const x=W*p[0],y=H*p[1],phase=t*1.5+index*1.7;
      glow(x,y,index%2?'125,202,219':'255,184,127',Math.min(W,H)*(.042+.007*Math.sin(phase)),.24+.045*Math.sin(phase));
      colorFill(x,y,1.3+Math.sin(phase)*.45,index%2?'168,230,249':'255,207,157',.7);
    });

    // Structure formation: filaments condense toward luminous haloes.
    ctx.save();
    for (let i=0;i<edges.length;i++) {
      const edge=edges[i],a=cosmicNodes[edge.i],b=cosmicNodes[edge.j];
      const shiftA=Math.sin(t*.53+a.t)*1.9,shiftB=Math.sin(t*.53+b.t)*1.9;
      ctx.beginPath();
      ctx.moveTo(a.x*W+shiftA,a.y*H);
      ctx.lineTo(b.x*W+shiftB,b.y*H);
      ctx.strokeStyle='rgba(87,188,207,'+edge.alpha.toFixed(3)+')';
      ctx.lineWidth=.7;ctx.stroke();
    }
    for (let i=0;i<cosmicNodes.length;i++) {
      const node=cosmicNodes[i];
      const x=node.x*W+Math.sin(t*.53+node.t)*1.9;
      const y=node.y*H+Math.cos(t*.43+node.t)*1.5;
      if (i % 14 === 0) glow(x,y,'255,181,118',Math.min(W,H)*.042,.20);
      colorFill(x,y,node.r,i%5===0?'255,191,126':'133,223,231',node.a*(.82+.18*Math.sin(t+node.t)));
    }
    ctx.restore();

    // Wide, gentle foreground starfield.
    for(let i=0;i<backgroundStars.length;i++){
      const s=backgroundStars[i],a=(.10+s.a*.37)*(.82+.18*Math.sin(t*1.5+s.phase));
      colorFill(s.x*W,s.y*H,s.size,'201,223,241',a);
    }

    // Light propagation / evolution ribbon from the hot era to cosmic web.
    ctx.save();ctx.globalCompositeOperation='screen';
    for(let layer=0;layer<3;layer++) {
      const grad=ctx.createLinearGradient(W*.16,0,W*.93,0);
      grad.addColorStop(0,'rgba(255,176,112,.21)');
      grad.addColorStop(.36,'rgba(151,196,226,.11)');
      grad.addColorStop(.72,'rgba(115,210,223,.13)');
      grad.addColorStop(1,'rgba(123,220,223,.03)');
      ctx.beginPath();
      for(let i=0;i<=120;i++) {
        const v=i/120, x=W*(.16+.78*v);
        const y=H*(.45+.032*Math.sin(v*13+t*.26+layer*.6)+layer*.025);
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      }
      ctx.lineWidth=layer===0?1.5:.6;
      ctx.strokeStyle=grad;
      ctx.stroke();
    }
    ctx.restore();

    // A drifting luminous packet advances through cosmic time.
    const phase=(t*.041)%1;
    const px=W*(.16+.78*phase);
    const py=H*(.45+.032*Math.sin(phase*13+t*.26));
    glow(px,py,phase<.26?'255,186,126':'152,235,231',Math.min(W,H)*.045,.19);
  }

  function resize() {
    const rect=canvas.getBoundingClientRect();
    if(!rect.width || !rect.height)return;
    width=rect.width;height=rect.height;
    canvasScale=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*canvasScale);
    canvas.height=Math.round(height*canvasScale);
    ctx.setTransform(canvasScale,0,0,canvasScale,0,0);
    drawBackground(0);
  }
  function updateEpoch(time) {
    const index=Math.floor((time*.041)%1 * epochNodes.length)%Math.max(1,epochNodes.length);
    if(index===lastEpoch)return;
    lastEpoch=index;
    epochNodes.forEach(function (node,i) { node.classList.toggle('current-epoch',i===index); });
  }
  function animate(now) {
    frameRequest=0;
    if(document.hidden||!visible||reducedMotion.matches)return;
    if(now-lastTime>40) {
      const time=now*.001;
      drawBackground(time);
      updateEpoch(time);
      lastTime=now;
    }
    frameRequest=window.requestAnimationFrame(animate);
  }
  function run() {
    if(frameRequest)window.cancelAnimationFrame(frameRequest);
    frameRequest=0;
    if(!document.hidden && visible && !reducedMotion.matches) {
      frameRequest=window.requestAnimationFrame(animate);
    } else {
      drawBackground(0);
      epochNodes.forEach(function (node) {node.classList.remove('current-epoch');});
    }
  }
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(function(entries){
      visible=!!entries[0] && entries[0].isIntersecting;
      run();
    },{threshold:.01});
    observer.observe(canvas);
  }
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',run);
  if(reducedMotion.addEventListener)reducedMotion.addEventListener('change',run);
  resize();
  run();
})();

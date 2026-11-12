'use strict';
/* Fast, source-attributed media with lazy loading and honest fallback.
 *
 * Hero: official IllustrisTNG TNG300 HD footage with opt-in 4K, plus local model.
 * AGN:  original self-hosted gas-density feedback cycle, lazy-loaded.
 * PopIII.1: exact author-provided Drive video, untouched.
 */
(() => {
  const root=document.body;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const heroVideo=document.getElementById('cosmos-real-video');
  const heroFallback=document.getElementById('cosmos-video-fallback');
  const heroSwitch=document.getElementById('cosmo-toggle-model');
  const qualitySwitch=document.getElementById('cosmo-quality');
  const heroHdUrl='https://www.tng-project.org/movies/tng/tng300_most_massive_cluster_gas_1080p.mp4';
  const hero4kUrl='https://www.tng-project.org/movies/tng/tng300_most_massive_cluster_gas_4k.mp4';
  const heroControls=[...document.querySelectorAll('.cosmo-model-controls')];
  const stageName=document.getElementById('story-name');
  const stageSub=document.getElementById('story-subtitle');
  const stageNo=document.getElementById('story-number');
  let heroIllustration=false,heroBackup=false,heroReady=false,heroQuality='HD';
  function playMedia(video){
    if(!video||document.hidden||reduced.matches)return;
    video.muted=true;video.defaultMuted=true;
    try{
      const p=video.play();
      if(p&&p.catch)p.catch(()=>{video.controls=true;});
    }catch(err){video.controls=true;}
  }
  function setHeroQuality(next){
    if(!heroVideo||heroBackup)return;
    const selected=next==='4K'?'4K':'HD';
    if(selected===heroQuality)return;
    const previousTime=Number.isFinite(heroVideo.currentTime)?heroVideo.currentTime:0;
    heroQuality=selected;
    heroReady=false;
    const source=heroVideo.querySelector('source');
    if(!source)return;
    source.src=selected==='4K'?hero4kUrl:heroHdUrl;
    heroVideo.load();
    if(previousTime>0){
      heroVideo.addEventListener('loadedmetadata',()=>{
        if(Number.isFinite(heroVideo.duration)&&heroVideo.duration>0){
          try{heroVideo.currentTime=Math.min(previousTime,Math.max(0,heroVideo.duration-.25));}catch(_){}
        }
      },{once:true});
    }
    if(qualitySwitch){
      qualitySwitch.textContent=selected==='4K'?'Switch to HD 1080p':'Play in 4K (larger file)';
      qualitySwitch.setAttribute('aria-pressed',String(selected==='4K'));
    }
    heroLabel();
    if(!heroIllustration)playMedia(heroVideo);
  }
  function heroLabel(){
    if(heroIllustration)return;
    if(stageName)stageName.textContent=heroBackup?'ILLUSTRIS · A VIRTUAL UNIVERSE':'TNG300 · SUPERCLUSTER FORMATION';
    if(stageSub)stageSub.textContent=heroBackup?'Illustris Collaboration · hydrodynamics, z = 127 → 0':'IllustrisTNG / D. Nelson · gas density · 500 Myr → present';
    if(stageNo)stageNo.textContent=heroBackup?'ILLUSTRIS':'TNG300 · '+(heroQuality==='4K'?'4K':'1080P');
    const metadata=document.getElementById('cosmo-epoch-readout');
    if(metadata)metadata.textContent=heroBackup?'Illustris · z 127 → 0':'TNG300 · gas density · '+(heroQuality==='4K'?'4K':'HD');
  }
  function enableHeroIllustration(enabled){
    heroIllustration=!!enabled;
    root.classList.toggle('cosmo-model-active',heroIllustration);
    heroControls.forEach(el=>{el.hidden=!heroIllustration;});
    if(heroSwitch){
      heroSwitch.setAttribute('aria-pressed',String(heroIllustration));
      heroSwitch.textContent=heroIllustration?'Show real simulation video':'Show illustrated CMB-to-today timeline';
    }
    if(heroVideo){
      heroVideo.hidden=heroIllustration||heroBackup;
      if(heroIllustration)heroVideo.pause();
      else if(!heroBackup)playMedia(heroVideo);
    }
    if(heroFallback){
      heroFallback.hidden=heroIllustration||!heroBackup;
      if(heroIllustration)heroFallback.removeAttribute('src');
      else if(heroBackup && !heroFallback.getAttribute('src'))heroFallback.src=heroFallback.dataset.embed;
    }
    if(heroIllustration){
      if(window.researchLoad)window.researchLoad('./cosmic-story.js?v=6')
        .then(()=>window.dispatchEvent(new Event('resize')))
        .catch(console.warn);
    }else heroLabel();
  }
  function heroBackupVideo(){
    if(heroBackup||heroIllustration)return;
    heroBackup=true;
    if(qualitySwitch)qualitySwitch.disabled=true;
    if(heroVideo){heroVideo.pause();heroVideo.hidden=true;}
    if(heroFallback){
      heroFallback.hidden=false;
      heroFallback.src=heroFallback.dataset.embed;
    }else enableHeroIllustration(true);
    heroLabel();
  }
  if(heroSwitch)heroSwitch.addEventListener('click',()=>enableHeroIllustration(!heroIllustration));
  if(qualitySwitch)qualitySwitch.addEventListener('click',()=>setHeroQuality(heroQuality==='HD'?'4K':'HD'));
  if(heroVideo){
    heroVideo.addEventListener('loadeddata',()=>{heroReady=true;heroLabel();});
    heroVideo.addEventListener('playing',()=>{heroReady=true;});
    heroVideo.addEventListener('error',()=>{
      if(heroQuality==='4K'){setHeroQuality('HD');}
      else heroBackupVideo();
    });
    // This timer runs only for the hero because it is visible at page load.
    window.setTimeout(()=>{
      if(!heroIllustration&&!heroReady&&heroVideo.readyState<2)heroBackupVideo();
    },18000);
    if(reduced.matches)heroVideo.controls=true;
  }
  if(heroFallback)heroFallback.addEventListener('error',()=>enableHeroIllustration(true));
  enableHeroIllustration(false);

  // The author-generated MP4 is the only AGN visualization. It remains unloaded
  // until near the viewport; the removed schematic is never fetched.
  const agnVideo=document.getElementById('agn-cycle-video');
  const agnError=document.getElementById('agn-video-error');
  let agnVisible=false,agnSourcesReady=false;
  function loadAGN(){
    if(!agnVideo||agnSourcesReady)return;
    agnSourcesReady=true;
    for(const source of agnVideo.querySelectorAll('source[data-src]')){
      source.src=source.dataset.src;
    }
    agnVideo.load();
  }
  if(agnVideo){
    agnVideo.addEventListener('error',()=>{
      if(agnError)agnError.hidden=false;
    });
    agnVideo.addEventListener('loadeddata',()=>{
      if(agnError)agnError.hidden=true;
    });
    const showAGN=()=>{
      agnVisible=true;
      loadAGN();
      playMedia(agnVideo);
    };
    if('IntersectionObserver' in window){
      const observer=new IntersectionObserver(entries=>{
        const near=!!entries[0]&&entries[0].isIntersecting;
        agnVisible=near;
        if(near)showAGN();
        else agnVideo.pause();
      },{rootMargin:'220px 0px',threshold:0});
      observer.observe(agnVideo);
    }else showAGN();
    if(reduced.matches)agnVideo.controls=true;
  }
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden){
      if(agnVideo)agnVideo.pause();
      if(heroVideo)heroVideo.pause();
    }else{
      if(heroVideo&&!heroIllustration&&!heroBackup)playMedia(heroVideo);
      if(agnVideo&&agnVisible)playMedia(agnVideo);
    }
  });

  const original=document.getElementById('popiii-original-iframe');
  if(original)original.setAttribute('aria-describedby','popiii-access-note');
})();

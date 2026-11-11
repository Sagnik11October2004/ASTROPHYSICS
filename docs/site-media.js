'use strict';
/* Fast, source-attributed media with lazy loading and honest fallback.
 *
 * Hero: official external cosmological footage, with optional local model.
 * AGN:  NASA Goddard quasar-wind feedback film only, deferred until near viewport.
 * PopIII.1: exact author-provided external Drive video, untouched.
 */
(() => {
  const root=document.body;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const heroVideo=document.getElementById('cosmos-real-video');
  const heroFallback=document.getElementById('cosmos-video-fallback');
  const heroSwitch=document.getElementById('cosmo-toggle-model');
  const heroControls=[...document.querySelectorAll('.cosmo-model-controls')];
  const stageName=document.getElementById('story-name');
  const stageSub=document.getElementById('story-subtitle');
  const stageNo=document.getElementById('story-number');
  let heroIllustration=false,heroBackup=false,heroReady=false,heroNear=true;
  function playMedia(video){
    if(!video||document.hidden||reduced.matches)return;
    video.muted=true;video.defaultMuted=true;
    try{
      const p=video.play();
      if(p&&p.catch)p.catch(()=>{video.controls=true;});
    }catch(err){video.controls=true;}
  }
  function heroLabel(){
    if(heroIllustration)return;
    if(stageName)stageName.textContent=heroBackup?'ILLUSTRIS · A VIRTUAL UNIVERSE':'GRAVITATIONAL STRUCTURE FORMATION';
    if(stageSub)stageSub.textContent=heroBackup?'Illustris Collaboration · hydrodynamics, z = 127 → 0':'NASA / MPE / Volker Springel · original simulation footage';
    if(stageNo)stageNo.textContent=heroBackup?'ILLUSTRIS':'NASA / MPE';
    const metadata=document.getElementById('cosmo-epoch-readout');
    if(metadata)metadata.textContent=heroBackup?'Illustris · z 127 → 0':'Real simulation footage';
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
    if(heroVideo){heroVideo.pause();heroVideo.hidden=true;}
    if(heroFallback){
      heroFallback.hidden=false;
      heroFallback.src=heroFallback.dataset.embed;
    }else enableHeroIllustration(true);
    heroLabel();
  }
  if(heroSwitch)heroSwitch.addEventListener('click',()=>enableHeroIllustration(!heroIllustration));
  if(heroVideo){
    heroVideo.addEventListener('loadeddata',()=>{heroReady=true;heroLabel();});
    heroVideo.addEventListener('playing',()=>{heroReady=true;});
    heroVideo.addEventListener('error',heroBackupVideo);
    // This timer runs only for the hero because it is visible at page load.
    window.setTimeout(()=>{
      if(!heroIllustration&&!heroReady&&heroVideo.readyState<2)heroBackupVideo();
    },11000);
    if(reduced.matches)heroVideo.controls=true;
  }
  if(heroFallback)heroFallback.addEventListener('error',()=>enableHeroIllustration(true));
  enableHeroIllustration(false);

  // The NASA film is now the only AGN visualization. It remains unloaded
  // until near the viewport; the removed schematic is never fetched.
  const agnVideo=document.getElementById('agn-nasa-video');
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

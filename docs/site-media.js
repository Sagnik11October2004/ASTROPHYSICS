'use strict';
/* Fast, source-attributed media with lazy loading and honest fallback.
 *
 * Hero: official external cosmological footage, with optional local model.
 * AGN:  NASA Goddard quasar-wind feedback film, deferred until in viewport;
 *       a separately labelled schematic illustrates *possible* recurrent fueling.
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

  const agnVideo=document.getElementById('agn-nasa-video');
  const agnCanvas=document.getElementById('agn-scene');
  const videoBtn=document.getElementById('agn-mode-video');
  const modelBtn=document.getElementById('agn-mode-model');
  const agnControls=[...document.querySelectorAll('.agn-model-controls')];
  const agnContainer=agnVideo?.closest('.astrolab-scene-wrap')||null;
  const agnLabel=document.getElementById('agn-phase');
  const agnSub=document.getElementById('agn-phase-sub');
  let agnModel=false,agnVisible=false,agnFetched=false,agnReady=false,agnStarted=false;
  const agnCredit={
    title:'NASA · QUASAR WIND AND GALACTIC OUTFLOW',
    subtitle:'Accretion-disk winds launch shocks and sweep gas out of the host galaxy'
  };
  function ensureAGN(){
    if(!agnVideo||agnFetched)return;
    agnFetched=true;
    for(const source of agnVideo.querySelectorAll('source[data-src]')){
      source.src=source.dataset.src;
    }
    agnVideo.load();
    if(reduced.matches)agnVideo.controls=true;
    window.setTimeout(()=>{
      if(agnFetched&&!agnReady&&agnVisible&&!agnModel&&agnVideo.readyState<2){
        useAGNModel(true);
        if(agnSub)agnSub.textContent='NASA video unavailable · showing the local illustrated cycle';
      }
    },13500);
  }
  function useAGNModel(value){
    agnModel=!!value;
    root.classList.toggle('agn-model-active',agnModel);
    root.classList.toggle('agn-video-active',!agnModel);
    if(agnContainer)agnContainer.classList.toggle('agn-model-active',agnModel);
    if(agnCanvas)agnCanvas.hidden=!agnModel;
    if(agnVideo){
      agnVideo.hidden=agnModel;
      if(agnModel)agnVideo.pause();
      else if(agnVisible){
        ensureAGN();
        playMedia(agnVideo);
      }
    }
    agnControls.forEach(el=>{el.hidden=!agnModel;});
    if(videoBtn){
      videoBtn.classList.toggle('active',!agnModel);
      videoBtn.setAttribute('aria-pressed',String(!agnModel));
    }
    if(modelBtn){
      modelBtn.classList.toggle('active',agnModel);
      modelBtn.setAttribute('aria-pressed',String(agnModel));
    }
    if(agnModel){
      if(window.researchLoad)window.researchLoad('./astrolab.js?v=6')
        .then(()=>window.dispatchEvent(new Event('resize')))
        .catch(console.warn);
    }else{
      if(agnLabel)agnLabel.textContent=agnCredit.title;
      if(agnSub)agnSub.textContent=agnCredit.subtitle;
    }
  }
  if(videoBtn)videoBtn.addEventListener('click',()=>useAGNModel(false));
  if(modelBtn)modelBtn.addEventListener('click',()=>useAGNModel(true));
  if(agnVideo){
    agnVideo.addEventListener('loadeddata',()=>{agnReady=true;});
    agnVideo.addEventListener('playing',()=>{agnReady=true;});
    agnVideo.addEventListener('error',()=>useAGNModel(true));
    const startVisible=()=>{agnVisible=true;if(!agnModel){ensureAGN();playMedia(agnVideo);}};
    if('IntersectionObserver' in window){
      const observer=new IntersectionObserver(entries=>{
        const entry=entries[0];
        const near=!!entry && entry.isIntersecting;
        if(near){startVisible();}
        else{
          agnVisible=false;
          agnVideo.pause();
        }
      },{rootMargin:'250px 0px',threshold:0});
      observer.observe(agnVideo);
    }else startVisible();
  }
  useAGNModel(false);
  // A background page should not continue decoding/rendering media.
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden){
      if(agnVideo)agnVideo.pause();
      if(heroVideo)heroVideo.pause();
    }else{
      if(heroVideo&&!heroIllustration&&!heroBackup)playMedia(heroVideo);
      if(agnVideo&&agnVisible&&!agnModel)playMedia(agnVideo);
    }
  });

  const original=document.getElementById('popiii-original-iframe');
  if(original)original.setAttribute('aria-describedby','popiii-access-note');
})();

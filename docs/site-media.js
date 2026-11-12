'use strict';
/* IllustrisTNG-only homepage hero, 1080p default / optional 4K.
 * No illustrated universe sequence or unrelated backup video.
 * The original self-hosted AGN cycle and Pop III player remain unchanged.
 */
(() => {
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const video=document.getElementById('cosmos-real-video');
  const qualityButton=document.getElementById('cosmo-quality');
  const playbackButton=document.getElementById('cosmo-playback');
  const errorNote=document.getElementById('cosmos-video-error');
  const stageNo=document.getElementById('story-number');
  const metadata=document.getElementById('cosmo-epoch-readout');
  const HD='https://www.tng-project.org/movies/tng/tng300_most_massive_cluster_gas_1080p.mp4';
  const UHD='https://www.tng-project.org/movies/tng/tng300_most_massive_cluster_gas_4k.mp4';
  let quality='HD',requestedPlayback=!reduced.matches,ready=false;
  function playMedia(el){
    if(!el||document.hidden||reduced.matches)return;
    el.muted=true;
    try{
      const p=el.play();
      if(p&&p.catch)p.catch(()=>{el.controls=true;updatePlayback();});
    }catch(_){el.controls=true;updatePlayback();}
  }
  function updatePlayback(){
    if(!playbackButton||!video)return;
    const active=!video.paused;
    playbackButton.textContent=active?'Pause video Ⅱ':'Play video ▶';
    playbackButton.setAttribute('aria-pressed',String(active));
  }
  function updateQuality(){
    if(qualityButton){
      qualityButton.textContent=quality==='4K'?'Switch to HD 1080p':'Play in 4K (larger file)';
      qualityButton.setAttribute('aria-pressed',String(quality==='4K'));
    }
    if(stageNo)stageNo.textContent=quality==='4K'?'TNG300 · 4K':'TNG300 · 1080P';
    if(metadata)metadata.textContent='TNG300 · gas density · '+quality;
  }
  function setQuality(next){
    if(!video||next===quality)return;
    const oldTime=Number.isFinite(video.currentTime)?video.currentTime:0;
    quality=next==='4K'?'4K':'HD';
    ready=false;
    if(errorNote)errorNote.hidden=true;
    const source=video.querySelector('source');
    if(!source)return;
    source.src=quality==='4K'?UHD:HD;
    video.load();
    if(oldTime>0){
      video.addEventListener('loadedmetadata',()=>{
        if(Number.isFinite(video.duration)&&video.duration>0){
          try{video.currentTime=Math.min(oldTime,video.duration-.2);}catch(_){}
        }
      },{once:true});
    }
    updateQuality();
    if(requestedPlayback)playMedia(video);
  }
  if(video){
    video.addEventListener('loadeddata',()=>{ready=true;if(errorNote)errorNote.hidden=true;});
    video.addEventListener('play',updatePlayback);
    video.addEventListener('pause',updatePlayback);
    video.addEventListener('error',()=>{
      if(quality==='4K')setQuality('HD');
      else{
        if(errorNote)errorNote.hidden=false;
        if(playbackButton)playbackButton.disabled=true;
      }
    });
    // If the source host blocks streaming, show a link rather than presenting
    // an unrelated illustrative or third-party movie as simulation footage.
    window.setTimeout(()=>{if(!ready&&video.readyState<2&&errorNote)errorNote.hidden=false;},18000);
    if(reduced.matches){video.autoplay=false;video.controls=true;video.pause();requestedPlayback=false;}
    updateQuality();updatePlayback();
  }
  if(qualityButton)qualityButton.addEventListener('click',()=>setQuality(quality==='HD'?'4K':'HD'));
  if(playbackButton)playbackButton.addEventListener('click',()=>{
    if(!video)return;
    if(video.paused){requestedPlayback=true;if(reduced.matches){video.play().catch(()=>{video.controls=true;});}else playMedia(video);}
    else{requestedPlayback=false;video.pause();}
    updatePlayback();
  });

  const agnVideo=document.getElementById('agn-cycle-video');
  const agnError=document.getElementById('agn-video-error');
  let agnVisible=false,agnSourcesReady=false;
  function loadAGN(){
    if(!agnVideo||agnSourcesReady)return;
    agnSourcesReady=true;
    for(const source of agnVideo.querySelectorAll('source[data-src]'))source.src=source.dataset.src;
    agnVideo.load();
  }
  if(agnVideo){
    agnVideo.addEventListener('error',()=>{if(agnError)agnError.hidden=false;});
    agnVideo.addEventListener('loadeddata',()=>{if(agnError)agnError.hidden=true;});
    const showAGN=()=>{agnVisible=true;loadAGN();playMedia(agnVideo);};
    if('IntersectionObserver' in window){
      new IntersectionObserver(entries=>{
        const near=!!entries[0]&&entries[0].isIntersecting;
        agnVisible=near;if(near)showAGN();else agnVideo.pause();
      },{rootMargin:'220px 0px',threshold:0}).observe(agnVideo);
    }else showAGN();
    if(reduced.matches)agnVideo.controls=true;
  }
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden){if(video)video.pause();if(agnVideo)agnVideo.pause();}
    else{
      if(video&&requestedPlayback)playMedia(video);
      if(agnVideo&&agnVisible)playMedia(agnVideo);
    }
  });
  const original=document.getElementById('popiii-original-iframe');
  if(original)original.setAttribute('aria-describedby','popiii-access-note');
})();
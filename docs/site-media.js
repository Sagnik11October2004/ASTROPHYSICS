'use strict';
/* The D. Nelson / IllustrisTNG homepage uses a small, self-hosted fast-start
 * H.264 movie and poster by default. Original 1080p and 4K remain opt-in.
 * No illustrative substitute footage. The original self-hosted AGN cycle and
 * Pop III.1 video remain unchanged.
 */
(() => {
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const video=document.getElementById('cosmos-real-video');
  const qualitySelect=document.getElementById('cosmo-quality');
  const playbackButton=document.getElementById('cosmo-playback');
  const speedSelect=document.getElementById('cosmo-speed');
  const errorNote=document.getElementById('cosmos-video-error');
  const loader=document.getElementById('cosmo-video-loader');
  const loaderText=document.getElementById('cosmo-loader-text');
  const stageNo=document.getElementById('story-number');
  const metadata=document.getElementById('cosmo-epoch-readout');
  const FAST='./media/tng300-supercluster-720p-faststart.mp4';
  const HD='https://www.tng-project.org/movies/tng/tng300_most_massive_cluster_gas_1080p.mp4';
  const UHD='https://www.tng-project.org/movies/tng/tng300_most_massive_cluster_gas_4k.mp4';
  const sources={FAST,HD,'4K':UHD};
  const labels={FAST:'720P FAST',HD:'1080P','4K':'4K'};
  let quality='FAST',requestedPlayback=!reduced.matches,ready=false;
  let pendingSeek=0,fastFallbackAttempted=false;
  const supportedSpeeds=[1,1.5,2,3];
  let playbackSpeed=2;
  function showLoader(message){
    if(loader)loader.hidden=false;
    if(loaderText&&message)loaderText.textContent=message;
  }
  function hideLoader(){
    if(loader)loader.hidden=true;
  }
  function updateQuality(){
    if(qualitySelect)qualitySelect.value=quality;
    if(stageNo)stageNo.textContent='TNG300 · '+labels[quality];
    if(metadata)metadata.textContent='TNG300 · gas density · '+labels[quality]+' · '+playbackSpeed+'×';
  }
  function setPlaybackSpeed(value){
    const next=Number(value);
    if(!supportedSpeeds.includes(next))return;
    playbackSpeed=next;
    if(video){
      video.defaultPlaybackRate=next;
      video.playbackRate=next;
    }
    if(speedSelect)speedSelect.value=String(next);
    updateQuality();
  }
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
  function setQuality(next){
    if(!video||!Object.prototype.hasOwnProperty.call(sources,next)||next===quality)return;
    pendingSeek=Number.isFinite(video.currentTime)&&video.currentTime>0?video.currentTime:0;
    quality=next;
    ready=false;
    if(errorNote)errorNote.hidden=true;
    if(playbackButton)playbackButton.disabled=false;
    showLoader(quality==='FAST'?'Loading optimized TNG300 footage…':'Loading original '+labels[quality]+' footage…');
    const source=video.querySelector('source');
    if(!source)return;
    source.src=sources[quality];
    updateQuality();
    video.load();
    // Native HTML5 playback-rate settings are re-applied on each load.
    setPlaybackSpeed(playbackSpeed);
    if(requestedPlayback)playMedia(video);
  }
  if(video){
    video.addEventListener('loadstart',()=>{
      ready=false;
      showLoader(quality==='FAST'?'Loading optimized TNG300 footage…':'Loading original '+labels[quality]+' footage…');
    });
    video.addEventListener('loadedmetadata',()=>{
      setPlaybackSpeed(playbackSpeed);
      if(pendingSeek>0&&Number.isFinite(video.duration)&&video.duration>.2){
        try{video.currentTime=Math.min(pendingSeek,video.duration-.2);}catch(_){}
        pendingSeek=0;
      }
    });
    video.addEventListener('loadeddata',()=>{
      ready=true;
      if(errorNote)errorNote.hidden=true;
      hideLoader();
      setPlaybackSpeed(playbackSpeed);
    });
    video.addEventListener('canplay',()=>{ready=true;hideLoader();});
    video.addEventListener('playing',()=>{hideLoader();updatePlayback();});
    video.addEventListener('waiting',()=>{
      if(!video.paused)showLoader('Buffering TNG300 video…');
    });
    video.addEventListener('play',updatePlayback);
    video.addEventListener('pause',updatePlayback);
    video.addEventListener('error',()=>{
      if(quality==='FAST'&&!fastFallbackAttempted){
        // Do not strand visitors if the locally hosted preview is unavailable.
        fastFallbackAttempted=true;
        setQuality('HD');
      }else if(quality==='4K'){
        setQuality('HD');
      }else{
        hideLoader();
        if(errorNote)errorNote.hidden=false;
        if(playbackButton)playbackButton.disabled=true;
      }
    });
    // A slow connection is not an error; keep the poster visible and show
    // loading status rather than reporting a false network failure.
    window.setTimeout(()=>{
      if(!ready&&video.readyState<2)showLoader('Still loading… choose a different quality if needed.');
    },10000);
    if(reduced.matches){
      video.autoplay=false;video.controls=true;video.pause();
      requestedPlayback=false;hideLoader();
    }
    setPlaybackSpeed(playbackSpeed);
    updatePlayback();
  }
  if(speedSelect)speedSelect.addEventListener('change',()=>setPlaybackSpeed(speedSelect.value));
  if(qualitySelect)qualitySelect.addEventListener('change',()=>setQuality(qualitySelect.value));
  if(playbackButton)playbackButton.addEventListener('click',()=>{
    if(!video)return;
    if(video.paused){
      requestedPlayback=true;
      if(reduced.matches){
        const p=video.play();if(p&&p.catch)p.catch(()=>{video.controls=true;});
      }else playMedia(video);
    }else{requestedPlayback=false;video.pause();}
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
  const original=document.getElementById('popiii-original-video');
  if(original)original.setAttribute('aria-describedby','popiii-access-note');
})();
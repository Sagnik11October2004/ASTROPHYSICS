'use strict';
/* Accessible fullscreen control for the owner's exact 1920 x 1032
 * Pop III.1 comparison video hosted on this same GitHub Pages site.
 * No frame cropping or video reconstruction.
 */
(() => {
  const button=document.getElementById('popiii-fullscreen');
  const wrapper=document.querySelector('#popiii .pop-video-frame');
  if(!button||!wrapper)return;
  const originalLabel=button.textContent;
  button.addEventListener('click',()=>{
    if(document.fullscreenElement===wrapper){
      if(typeof document.exitFullscreen==='function'){
        const result=document.exitFullscreen();
        if(result&&typeof result.catch==='function')result.catch(()=>{});
      }
      return;
    }
    if(typeof wrapper.requestFullscreen!=='function'){
      button.textContent='Fullscreen unavailable · open MP4 directly';
      return;
    }
    try{
      const result=wrapper.requestFullscreen();
      if(result&&typeof result.catch==='function'){
        result.catch(()=>{
          button.textContent='Fullscreen unavailable · use Open in Drive';
        });
      }
    }catch(_){
      button.textContent='Fullscreen unavailable · use Open in Drive';
    }
  });
  document.addEventListener('fullscreenchange',()=>{
    if(document.fullscreenElement===wrapper){
      button.textContent='Exit fullscreen ⤢';
      button.setAttribute('aria-pressed','true');
    }else{
      button.textContent=originalLabel;
      button.setAttribute('aria-pressed','false');
    }
  });
  button.setAttribute('aria-pressed','false');
})();
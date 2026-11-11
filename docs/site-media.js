'use strict';
/* External-media safeguards for the user's research portfolio.
   Videos are original science footage attributed to their providers.
   Private Drive media is never made public programmatically. */
(() => {
 const root=document.body;
 const cosmosVideo=document.getElementById('cosmos-real-video');
 const cosmosIFrame=document.getElementById('cosmos-video-fallback');
 const cosmosButton=document.getElementById('cosmo-toggle-model');
 const cosmosControls=[...document.querySelectorAll('.cosmo-model-controls')];
 const stageName=document.getElementById('story-name');
 const stageSub=document.getElementById('story-subtitle');
 const stageNumber=document.getElementById('story-number');
 let cosmosModel=false, backupRequested=false, primaryReady=false;
 const textForVideo={
   name:'GRAVITATIONAL STRUCTURE FORMATION',
   subtitle:'NASA / MPE / Volker Springel · simulation footage',
   number:'RESEARCH VIDEO'
 };
 function playVideo(video){
   if(!video)return;
   video.muted=true;video.defaultMuted=true;
   try{
     const p=video.play();
     if(p&&typeof p.catch==='function')p.catch(function(){video.controls=true;});
   }catch(error){video.controls=true;}
 }
 function cosmosLabel(){
   if(cosmosModel)return;
   if(stageName)stageName.textContent=backupRequested?'ILLUSTRIS · A VIRTUAL UNIVERSE':textForVideo.name;
   if(stageSub)stageSub.textContent=backupRequested?
     'Illustris Collaboration · cosmological hydrodynamics, z = 127 → 0':
     textForVideo.subtitle;
   if(stageNumber)stageNumber.textContent=backupRequested?'ILLUSTRIS':'NASA / MPE';
 }
 function setCosmosModel(value){
   cosmosModel=!!value;
   root.classList.toggle('cosmo-model-active',cosmosModel);
   cosmosControls.forEach(function(el){el.hidden=!cosmosModel;});
   if(cosmosButton){
     cosmosButton.setAttribute('aria-pressed',String(cosmosModel));
     cosmosButton.textContent=cosmosModel?'Show real simulation video':'Show illustrated CMB-to-today timeline';
   }
   if(cosmosVideo){cosmosVideo.hidden=cosmosModel||backupRequested; if(cosmosModel)cosmosVideo.pause();else if(!backupRequested)playVideo(cosmosVideo);}
   if(cosmosIFrame){
     cosmosIFrame.hidden=cosmosModel||!backupRequested;
     if(cosmosModel)cosmosIFrame.removeAttribute('src');
     else if(backupRequested&&!cosmosIFrame.getAttribute('src'))cosmosIFrame.src=cosmosIFrame.dataset.embed;
   }
   if(cosmosModel){
     window.dispatchEvent(new Event('resize'));
   }else cosmosLabel();
 }
 function useCosmosBackup(){
   if(backupRequested||cosmosModel)return;
   backupRequested=true;
   if(cosmosVideo){cosmosVideo.pause();cosmosVideo.hidden=true;}
   if(cosmosIFrame){
     cosmosIFrame.hidden=false;
     cosmosIFrame.src=cosmosIFrame.dataset.embed;
   }else setCosmosModel(true);
   cosmosLabel();
 }
 if(cosmosButton)cosmosButton.addEventListener('click',()=>setCosmosModel(!cosmosModel));
 if(cosmosVideo){
   cosmosVideo.addEventListener('loadeddata',()=>{primaryReady=true;cosmosLabel();});
   cosmosVideo.addEventListener('playing',()=>{primaryReady=true;});
   cosmosVideo.addEventListener('error',useCosmosBackup);
   cosmosVideo.addEventListener('stalled',()=>{
     if(cosmosVideo.readyState<2)window.setTimeout(()=>{
       if(!primaryReady&&cosmosVideo.readyState<2)useCosmosBackup();
     },5000);
   });
   window.setTimeout(()=>{
     if(!primaryReady&&cosmosVideo.readyState<2&&!cosmosModel)useCosmosBackup();
   },12000);
 }
 if(cosmosIFrame)cosmosIFrame.addEventListener('error',()=>{setCosmosModel(true);});
 setCosmosModel(false);

 // Distinguish an actual jet-feedback simulation from a conceptual duty cycle.
 const agnVideo=document.getElementById('agn-nasa-video');
 const agnCanvas=document.getElementById('agn-scene');
 const modeVideo=document.getElementById('agn-mode-video');
 const modeModel=document.getElementById('agn-mode-model');
 const controls=[...document.querySelectorAll('.agn-model-controls')];
 const agnContainer=agnVideo?.closest('.astrolab-scene-wrap')||null;
 let agnModel=false,agnReady=false;
 function setAGNModel(value){
   agnModel=!!value;
   root.classList.toggle('agn-model-active',agnModel);
   root.classList.toggle('agn-video-active',!agnModel);
   if(agnContainer)agnContainer.classList.toggle('agn-model-active',agnModel);
   if(agnCanvas)agnCanvas.hidden=!agnModel;
   if(agnVideo){
     agnVideo.hidden=agnModel;
     if(agnModel)agnVideo.pause();
     else playVideo(agnVideo);
   }
   controls.forEach(el=>{el.hidden=!agnModel;});
   if(modeVideo){
     modeVideo.classList.toggle('active',!agnModel);
     modeVideo.setAttribute('aria-pressed',String(!agnModel));
   }
   if(modeModel){
     modeModel.classList.toggle('active',agnModel);
     modeModel.setAttribute('aria-pressed',String(agnModel));
   }
   if(agnModel)window.dispatchEvent(new Event('resize'));
 }
 if(modeVideo)modeVideo.addEventListener('click',()=>setAGNModel(false));
 if(modeModel)modeModel.addEventListener('click',()=>setAGNModel(true));
 if(agnVideo){
   agnVideo.addEventListener('loadeddata',()=>{agnReady=true;});
   agnVideo.addEventListener('playing',()=>{agnReady=true;});
   agnVideo.addEventListener('error',()=>setAGNModel(true));
   window.setTimeout(()=>{
     if(!agnReady&&agnVideo.readyState<2&&!agnModel)setAGNModel(true);
   },12500);
 }
 setAGNModel(false);

 // Provide an accessible escape hatch if the owner's private Drive iframe is blocked.
 const popFrame=document.getElementById('popiii-original-iframe');
 if(popFrame){
   popFrame.setAttribute('aria-describedby','popiii-access-note');
 }
})();

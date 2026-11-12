'use strict';
/* Lazy research modules. Keep the landing page fast by avoiding off-screen
 * Monte Carlo simulation, WebGL compilation and illustrated-AGN initialization. */
(() => {
  const scripts=new Map();
  function load(src){
    if(scripts.has(src))return scripts.get(src);
    const promise=new Promise((resolve,reject)=>{
      const el=document.createElement('script');
      el.src=src;el.async=true;
      el.addEventListener('load',resolve,{once:true});
      el.addEventListener('error',()=>reject(new Error('Unable to load '+src)),{once:true});
      document.head.appendChild(el);
    });
    scripts.set(src,promise);
    return promise;
  }
  window.researchLoad=load;
  function near(id,src,margin){
    const el=document.getElementById(id);
    if(!el)return;
    if(!('IntersectionObserver' in window)){
      load(src).catch(console.warn);
      return;
    }
    const observer=new IntersectionObserver((entries)=>{
      if(entries.some(entry=>entry.isIntersecting)){
        observer.disconnect();
        load(src).catch(err=>{
          const outer=el.closest('.astrolab-scene-wrap');
          if(outer)outer.classList.add('research-load-failed');
          console.warn(err);
        });
      }
    },{rootMargin:margin,threshold:0});
    observer.observe(el);
  }
  // Loading the MCRT physics engine is the expensive, interactive part;
  // only start the engine when its section is approaching the viewport.
  near('mcrt-scene','./mcrt-single.js?v=6','480px 0px');
  // WebGL program initialization is unnecessary until the portrait/profile scene.
  near('blackhole-canvas','./bh-relativistic.js?v=7','240px 0px');
})();

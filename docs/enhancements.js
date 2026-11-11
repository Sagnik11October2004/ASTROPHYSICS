'use strict';
/* Progressive enhancements; all core content remains readable without JS. */
(function(){
  document.querySelectorAll('.mentor-photo').forEach(function(photo){
    function fallback(){
      photo.style.display='none';
      const frame=photo.closest('.mentor-avatar');
      if(frame) frame.classList.add('portrait-unavailable');
    }
    photo.addEventListener('error',fallback,{once:true});
    if(photo.complete&&photo.naturalWidth===0)fallback();
    photo.setAttribute('title','Institutional / academic portrait. Open the linked supervisor name for the source profile.');
  });
  // Named landmark navigation for keyboard and screen-reader users.
  document.querySelectorAll('.cv-project-card').forEach(function(card,i){
    const title=card.querySelector('h3');
    if(!title)return;
    title.id='project-'+(i+1);
    card.setAttribute('aria-labelledby',title.id);
  });
})();

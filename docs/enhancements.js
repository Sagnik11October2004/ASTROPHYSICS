'use strict';
/* Progressive enhancements. Never silently break a mentor portrait. */
(function(){
  document.querySelectorAll('.mentor-photo').forEach(function(photo){
    const alternates=(photo.getAttribute('data-photo-alternates')||'')
      .split('|').map(function(v){return v.trim();}).filter(Boolean);
    let index=0;
    function attemptNext(){
      if(index<alternates.length){
        const next=alternates[index++];
        if(photo.src===next){attemptNext();return;}
        photo.style.display='block';
        photo.src=next;
      }else{
        photo.style.display='none';
        const frame=photo.closest('.mentor-avatar');
        if(frame)frame.classList.add('portrait-unavailable');
      }
    }
    photo.addEventListener('error',attemptNext);
    if(photo.complete&&photo.naturalWidth===0)attemptNext();
    photo.setAttribute('title','Research mentor portrait from an institutional or public professional profile.');
  });
  document.querySelectorAll('.cv-project-card').forEach(function(card,i){
    const title=card.querySelector('h3');
    if(!title)return;
    title.id='project-'+(i+1);
    card.setAttribute('aria-labelledby',title.id);
  });
})();

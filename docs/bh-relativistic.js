'use strict';
/* Approximate relativistic accretion-disk image for an educational website.
   This is an interactive GPU-shaded visualization, not a Kerr spacetime GRRT code.
   We approximate Doppler boosting, inward-rising disk temperature and a lensed
   photon-ring motif. It should not be interpreted as a scientific image. */
(function(){
  const existing=document.getElementById('blackhole-canvas');
  const slider=document.getElementById('bh-inclination');
  const readout=document.getElementById('bh-angle');
  if(!existing||!existing.parentNode)return;
  const canvas=document.createElement('canvas');
  canvas.id='bh-webgl';
  canvas.setAttribute('role','img');
  canvas.setAttribute('aria-label','Animated GPU-shaded black-hole accretion disk with inclination control, approximate Doppler boosting and photon-ring-inspired light bending');
  existing.parentNode.insertBefore(canvas,existing.nextSibling);
  const gl=canvas.getContext('webgl',{alpha:true,antialias:false,preserveDrawingBuffer:false,powerPreference:'low-power'});
  if(!gl){canvas.remove();return;}
  const vertex=`
    attribute vec2 aPos;
    void main() { gl_Position=vec4(aPos,0.0,1.0); }
  `;
  const fragment=`
    precision highp float;
    uniform vec2 uResolution;
    uniform float uTime;
    uniform float uInc;
    float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float noise(vec2 p){
      vec2 i=floor(p),f=fract(p);
      f=f*f*(3.0-2.0*f);
      return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);
    }
    float fbm(vec2 p){
      float f=0.,a=.5;
      for(int i=0;i<4;i++){f+=a*noise(p);p=p*2.01+vec2(2.3,4.7);a*=.5;}
      return f;
    }
    void main(){
      vec2 pos=(gl_FragCoord.xy-uResolution*.5)/min(uResolution.x,uResolution.y)*2.52;
      float cr=cos(.19),sr=sin(.19);
      vec2 uv=vec2(cr*pos.x-sr*pos.y,sr*pos.x+cr*pos.y);
      float b=length(uv);
      float incl=radians(uInc);
      float ci=max(.17,cos(incl));
      float rr=length(vec2(uv.x,uv.y/ci));
      float phi=atan(uv.y/ci,uv.x);
      vec3 col=vec3(.003,.009,.018);
      // Weak gaseous background + deterministic pinprick stars.
      vec2 starCell=floor(gl_FragCoord.xy/12.);
      float starSeed=hash(starCell);
      float starShape=length(fract(gl_FragCoord.xy/12.)-.5);
      float twinkle=.7+.3*sin(uTime*.7+starSeed*26.);
      col+=vec3(.19,.3,.43)*step(.989,starSeed)*exp(-starShape*95.)*twinkle;
      float nebulosity=fbm(uv*3.+vec2(.32,-.5));
      col+=vec3(.008,.025,.045)*nebulosity;
      // A cool, geometrically thin annulus with viscous-like temperature scaling.
      float rin=.342,rout=1.07;
      float annulus=smoothstep(rin,rin+.045,rr)*(1.-smoothstep(rout-.12,rout,rr));
      float t=pow(max(rr,.34)/rin,-.75)*pow(max(.002,1.-sqrt(rin/max(rr,rin+.005))),.25);
      t=clamp(t*1.8,0.,1.);
      float spiral=sin(phi*11.5-1.8*uTime+10.5*log(max(rr,.1)));
      float filaments=fbm(vec2(phi*10.+.12*uTime,rr*56.-1.9*uTime));
      float turbulence=.58+.25*spiral+.5*filaments;
      float approach=-uv.x/max(rr,.2); // left side is beamed toward the viewer
      float beta=clamp(.60*sqrt(rin/max(rr,rin)),.2,.71);
      float doppler=sqrt(1.-beta*beta)/(1.-beta*sin(incl)*approach);
      float boost=pow(doppler,3.1);
      float gravShift=sqrt(max(.3,1.-.16/max(rr,.17)));
      float bright=annulus*(.12+1.5*t)*max(.12,turbulence)*boost*gravShift;
      vec3 hot=vec3(1.25,1.37,1.45);
      vec3 warm=vec3(1.18,.48,.16);
      vec3 amber=vec3(.64,.19,.06);
      vec3 disk=mix(amber,warm,smoothstep(.15,.5,t));
      disk=mix(disk,hot,smoothstep(.49,.85,t));
      // Beamed, hotter patches trend white-blue while receding sectors redden.
      disk=mix(disk,vec3(.64,.82,1.2),clamp((boost-1.4)*.2,0.,.35));
      vec3 emission=disk*bright;
      // Back half, being farther from the viewer, is dimmer; front half is hotter.
      emission*=uv.y<0.?.76:1.15;
      // Horizon/shadow mask; disk can pass across the foreground at the lower edge.
      float shadow=1.-smoothstep(.24,.282,b);
      col+=emission*(1.-shadow);
      // Strong-field bending approximation: thin luminous secondary arcs above/below.
      float arch=.297+.125*sqrt(max(0.,1.-pow(uv.x/.60,2.)));
      float upperArc=exp(-pow((uv.y-arch)*65.,2.))*(1.-smoothstep(.47,.62,abs(uv.x)));
      float lowerArc=exp(-pow((uv.y+arch)*63.,2.))*(1.-smoothstep(.44,.62,abs(uv.x)));
      float arcNoise=.72+.28*sin(uv.x*39.+uTime*.5);
      col+=vec3(1.3,.66,.28)*(upperArc*.72+lowerArc*.15)*arcNoise;
      // Primary ring: enhanced brightness on approach side, with narrow faint halo.
      float ring=exp(-pow((b-.278)*88.,2.));
      float ringHalo=exp(-pow((b-.29)*29.,2.));
      float beamSide=1.4-.7*smoothstep(-.3,.3,uv.x);
      col+=vec3(1.28,.79,.41)*ring*beamSide*.80;
      col+=vec3(.27,.11,.06)*ringHalo*.26;
      col*=1.-shadow*.98;
      col+=vec3(1.28,.86,.46)*ring*beamSide*.64;
      // Larger luminous disc; emission can show in foreground below the horizon.
      float front=step(.03,uv.y)*annulus;
      col+=emission*front*.12;
      // Tone mapping, colour grading and gentle vignette.
      col=vec3(1.)-exp(-max(col,vec3(0.))*1.35);
      float vignette=1.-.35*smoothstep(.65,1.8,length(pos));
      col*=vignette;
      col=pow(col,vec3(.91));
      gl_FragColor=vec4(col,1.);
    }
  `;
  function compile(type,source){
    const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){
      console.warn('Black-hole GPU shader unavailable:',gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);return null;
    }
    return shader;
  }
  const vs=compile(gl.VERTEX_SHADER,vertex),fs=compile(gl.FRAGMENT_SHADER,fragment);
  if(!vs||!fs){canvas.remove();return;}
  const prog=gl.createProgram();
  gl.attachShader(prog,vs);gl.attachShader(prog,fs);gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog,gl.LINK_STATUS)){
    console.warn('GPU program unavailable:',gl.getProgramInfoLog(prog));
    canvas.remove();return;
  }
  gl.useProgram(prog);
  const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
  const pos=gl.getAttribLocation(prog,'aPos');
  gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
  const uResolution=gl.getUniformLocation(prog,'uResolution'),uTime=gl.getUniformLocation(prog,'uTime'),uInc=gl.getUniformLocation(prog,'uInc');
  existing.classList.add('bh-fallback-hidden');
  let visible=true,handle=0,lastTime=0;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  function resize(){
    const box=existing.getBoundingClientRect();
    const scale=Math.min(devicePixelRatio||1,1.5);
    canvas.width=Math.max(64,Math.round(box.width*scale));
    canvas.height=Math.max(64,Math.round(box.height*scale));
    gl.viewport(0,0,canvas.width,canvas.height);
    render(0);
  }
  function render(time){
    gl.useProgram(prog);
    gl.uniform2f(uResolution,canvas.width,canvas.height);
    gl.uniform1f(uTime,time);
    gl.uniform1f(uInc,Number(slider?slider.value:65));
    gl.drawArrays(gl.TRIANGLES,0,6);
  }
  function tick(now){
    handle=0;
    if(document.hidden||!visible||reduced.matches)return;
    if(now-lastTime>42){render(now*.001);lastTime=now;}
    handle=requestAnimationFrame(tick);
  }
  function sync(){
    if(handle)cancelAnimationFrame(handle);handle=0;
    if(!document.hidden&&visible&&!reduced.matches)handle=requestAnimationFrame(tick);
    else render(0);
  }
  if(slider)slider.addEventListener('input',function(){
    if(readout)readout.textContent=slider.value+'°';
    render(performance.now()*.001);
  });
  if('IntersectionObserver' in window){
    new IntersectionObserver(function(entries){visible=!!entries[0]&&entries[0].isIntersecting;sync();},{threshold:.02}).observe(canvas);
  }
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',sync);
  if(reduced.addEventListener)reduced.addEventListener('change',sync);
  canvas.addEventListener('webglcontextlost',function(event){event.preventDefault();existing.classList.remove('bh-fallback-hidden');canvas.style.display='none';});
  resize();sync();
})();

'use strict';
/* A small, independent browser adaptation of the methods and initial conditions
 * in Sagnik11October2004/REAL_FLUID (GPL-3.0):
 * - cell-centered signed-distance level set, phi < 0 means liquid
 * - staggered (MAC) velocity field with semi-Lagrangian backtracing
 * - gravity and optional curvature-localized capillary forcing
 * - free-surface pressure projection (Gauss-Seidel/SOR rather than source PCG)
 * - viscosity, level-set redistancing, and modest volume correction
 *
 * The original C++/OpenGL executable is NOT run in the visitor's browser.
 * The bubble case is an experimental extension beyond its three original ICs.
 * This demonstration is for physical intuition, not quantitative benchmark data.
 */
(() => {
  const canvas=document.getElementById('levelset-canvas');
  if(!canvas)return;
  const ctx=canvas.getContext('2d',{alpha:false});
  if(!ctx)return;
  const q=id=>document.getElementById(id);
  const run=q('levelset-run'),stepButton=q('levelset-step'),reset=q('levelset-reset');
  const speed=q('levelset-speed'),gravity=q('levelset-gravity'),sigma=q('levelset-sigma');
  const vectors=q('levelset-vectors'),grid=q('levelset-grid'),correction=q('levelset-volume-correct');
  const gravityOut=q('levelset-gravity-value'),sigmaOut=q('levelset-sigma-value');
  const timeOut=q('levelset-time'),areaOut=q('levelset-area'),divOut=q('levelset-divergence');
  const status=q('levelset-status'),description=q('levelset-description');
  const tabs=Array.from(document.querySelectorAll('[data-levelset-case]'));
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const N=64,H=1/N,DT=0.006;
  const clip=(x,a,b)=>Math.max(a,Math.min(b,x));
  const clamp01=x=>clip(x,0,1);
  const scenarios={
    dam:{title:'DAM BREAK',description:'A liquid column initially occupies x < 0.4 and y < 0.6. Gravity collapses its free surface, while the projection step restricts divergence inside the liquid.'},
    drop:{title:'DROPLET IMPACT',description:'A drop centered near (0.5, 0.8), radius 0.15, falls toward a shallow y < 0.2 pool. This is the original drop-into-liquid initial condition, not a rising gas bubble.'},
    equilibrium:{title:'FREE-SURFACE EQUILIBRIUM',description:'A thin layer, with a small wave perturbation added to make settling visible. The original C++ flat-layer initial condition is y < 0.1.'},
    bubble:{title:'BUBBLE / CAVITY · EXPERIMENTAL',description:'An added gas cavity inside a deeper liquid column. Its rise and deformation are qualitative; the original C++ source does not contain a fully resolved two-fluid buoyant bubble model.'}
  };
  function bilinear(a,w,h,x,y){
    x=clip(x,0,w-1.000001);y=clip(y,0,h-1.000001);
    const i=Math.floor(x),j=Math.floor(y),fx=x-i,fy=y-j,p=i+j*w;
    return (a[p]*(1-fx)+a[p+1]*fx)*(1-fy)
          +(a[p+w]*(1-fx)+a[p+w+1]*fx)*fy;
  }
  class Solver {
    constructor(mode){
      this.n=N;this.mode=mode;this.phi=new Float32Array(N*N);
      this.u=new Float32Array((N+1)*N);this.v=new Float32Array(N*(N+1));
      this.p=new Float32Array(N*N);this.advU=new Float32Array(this.u.length);
      this.advV=new Float32Array(this.v.length);this.nextPhi=new Float32Array(N*N);
      this.curvature=new Float32Array(N*N);this.age=0;this.steps=0;
      this.gravity=9.8;this.sigma=.007;this.viscosity=.00012;this.volumeCorrect=true;
      this.initialize(mode);this.initialArea=this.area();this.divergence=0;
    }
    U(i,j){return i+(N+1)*j;} V(i,j){return i+N*j;} I(i,j){return i+N*j;}
    phiAt(x,y){return bilinear(this.phi,N,N,x*N-.5,y*N-.5);}
    sampleU(x,y,a=this.u){return bilinear(a,N+1,N,x*N,y*N-.5);}
    sampleV(x,y,a=this.v){return bilinear(a,N,N+1,x*N-.5,y*N);}
    velocity(x,y,uu=this.u,vv=this.v){return [this.sampleU(x,y,uu),this.sampleV(x,y,vv)];}
    initialize(mode){
      this.mode=mode;
      for(let j=0;j<N;j++)for(let i=0;i<N;i++){
        const x=(i+.5)*H,y=(j+.5)*H;
        let d;
        switch(mode){
          case 'drop':d=Math.min(y-.20,Math.hypot(x-.50,y-.80)-.15);break;
          case 'equilibrium':d=y-(.10+.035*Math.cos(2*Math.PI*x));break;
          case 'bubble':d=Math.max(y-.87,.115-Math.hypot(x-.50,y-.35));break;
          default:d=Math.max(x-.4,y-.60);break;
        }
        this.phi[this.I(i,j)]=d;
      }
      this.u.fill(0);this.v.fill(0);this.p.fill(0);this.age=0;this.steps=0;
      this.reinitialize();
      this.initialArea=this.area();this.divergence=0;
    }
    area(){
      const e=H*.8;let sum=0;
      for(let k=0;k<this.phi.length;k++)
        sum+=clamp01(.5-this.phi[k]/(2*e));
      return sum/(N*N);
    }
    // Approximate 2D signed-distance reconstruction with four sweep directions.
    // The C++ source instead uses a geometric redistancing / marching method.
    reinitialize(){
      const phi=this.phi,n=N,sign=new Int8Array(n*n);
      const dist=new Float32Array(n*n);
      dist.fill(1e6);
      for(let j=0;j<n;j++)for(let i=0;i<n;i++){
        const k=this.I(i,j),f=phi[k];sign[k]=f<0?-1:1;
        if(i>0){
          const nb=k-1,ff=phi[nb];
          if((f<0)!==(ff<0))dist[k]=Math.min(dist[k],H*Math.abs(f)/(Math.abs(f)+Math.abs(ff)+1e-12));
        }
        if(i<n-1){
          const nb=k+1,ff=phi[nb];
          if((f<0)!==(ff<0))dist[k]=Math.min(dist[k],H*Math.abs(f)/(Math.abs(f)+Math.abs(ff)+1e-12));
        }
        if(j>0){
          const nb=k-n,ff=phi[nb];
          if((f<0)!==(ff<0))dist[k]=Math.min(dist[k],H*Math.abs(f)/(Math.abs(f)+Math.abs(ff)+1e-12));
        }
        if(j<n-1){
          const nb=k+n,ff=phi[nb];
          if((f<0)!==(ff<0))dist[k]=Math.min(dist[k],H*Math.abs(f)/(Math.abs(f)+Math.abs(ff)+1e-12));
        }
      }
      for(let sweep=0;sweep<3;sweep++){
        for(let pass=0;pass<4;pass++){
          const di=(pass&1)?-1:1,dj=(pass&2)?-1:1;
          for(let yy=0;yy<n;yy++){
            const j=dj>0?yy:n-1-yy;
            for(let xx=0;xx<n;xx++){
              const i=di>0?xx:n-1-xx,k=i+j*n;
              const ax=Math.min(i>0?dist[k-1]:1e6,i<n-1?dist[k+1]:1e6);
              const ay=Math.min(j>0?dist[k-n]:1e6,j<n-1?dist[k+n]:1e6);
              const diff=Math.abs(ax-ay);
              const candidate=diff>=H?Math.min(ax,ay)+H:
                .5*(ax+ay+Math.sqrt(Math.max(0,2*H*H-diff*diff)));
              if(candidate<dist[k])dist[k]=candidate;
            }
          }
        }
      }
      for(let k=0;k<phi.length;k++)
        phi[k]=sign[k]*(dist[k]<1e5?dist[k]:.5);
    }
    updateCurvature(){
      const f=this.phi,kappa=this.curvature;
      kappa.fill(0);
      const ih=1/H,ih2=ih*ih;
      for(let j=1;j<N-1;j++)for(let i=1;i<N-1;i++){
        const k=this.I(i,j);
        const x=.5*(f[k+1]-f[k-1])*ih,y=.5*(f[k+N]-f[k-N])*ih;
        const xx=(f[k+1]+f[k-1]-2*f[k])*ih2;
        const yy=(f[k+N]+f[k-N]-2*f[k])*ih2;
        const xy=.25*(f[k+N+1]-f[k+N-1]-f[k-N+1]+f[k-N-1])*ih2;
        const g=x*x+y*y;
        if(g>.05)kappa[k]=clip((xx*y*y-2*xy*x*y+yy*x*x)/(Math.pow(g,1.5)), -35,35);
      }
    }
    advectVelocity(){
      const oldU=this.u,oldV=this.v,nu=this.advU,nv=this.advV;
      const velo=(x,y)=>[this.sampleU(x,y,oldU),this.sampleV(x,y,oldV)];
      for(let j=0;j<N;j++)for(let i=0;i<=N;i++){
        const x=i*H,y=(j+.5)*H,k=this.U(i,j);
        if(i===0||i===N){nu[k]=0;continue;}
        const a=velo(x,y),b=velo(x-.5*DT*a[0],y-.5*DT*a[1]);
        nu[k]=this.sampleU(x-DT*b[0],y-DT*b[1],oldU);
      }
      for(let j=0;j<=N;j++)for(let i=0;i<N;i++){
        const x=(i+.5)*H,y=j*H,k=this.V(i,j);
        if(j===0||j===N){nv[k]=0;continue;}
        const a=velo(x,y),b=velo(x-.5*DT*a[0],y-.5*DT*a[1]);
        nv[k]=this.sampleV(x-DT*b[0],y-DT*b[1],oldV);
      }
      const viscosityFactor=this.viscosity*DT/(H*H);
      for(let j=1;j<N-1;j++)for(let i=1;i<N;i++){
        const k=this.U(i,j);
        this.u[k]=nu[k]+viscosityFactor*(nu[k-1]+nu[k+1]+nu[k-(N+1)]+nu[k+(N+1)]-4*nu[k]);
      }
      for(let j=1;j<N;j++)for(let i=1;i<N-1;i++){
        const k=this.V(i,j);
        this.v[k]=nv[k]+viscosityFactor*(nv[k-1]+nv[k+1]+nv[k-N]+nv[k+N]-4*nv[k]);
      }
      for(let j=0;j<N;j++){this.u[this.U(0,j)]=0;this.u[this.U(N,j)]=0;}
      for(let i=0;i<N;i++){this.v[this.V(i,0)]=0;this.v[this.V(i,N)]=0;}
    }
    applyForces(){
      const phi=this.phi,sigma=this.sigma,delta=H*1.5,curv=this.curvature;
      if(sigma>0)this.updateCurvature();
      for(let j=0;j<N;j++)for(let i=1;i<N;i++){
        const a=this.I(i-1,j),b=this.I(i,j),k=this.U(i,j);
        const wa=phi[a]<0,wb=phi[b]<0;
        if(!(wa||wb)){this.u[k]=0;continue;}
        if(sigma && Math.abs(phi[a]+phi[b])*.5<delta){
          const force=.5*(curv[a]+curv[b])*(phi[b]-phi[a])/H;
          this.u[k]+=DT*sigma*clip(force,-55,55);
        }
      }
      for(let j=1;j<N;j++)for(let i=0;i<N;i++){
        const a=this.I(i,j-1),b=this.I(i,j),k=this.V(i,j);
        const wa=phi[a]<0,wb=phi[b]<0;
        if(!(wa||wb)){this.v[k]=0;continue;}
        this.v[k]-=DT*this.gravity;
        if(sigma && Math.abs(phi[a]+phi[b])*.5<delta){
          const force=.5*(curv[a]+curv[b])*(phi[b]-phi[a])/H;
          this.v[k]+=DT*sigma*clip(force,-55,55);
        }
      }
    }
    project(){
      const phi=this.phi,p=this.p,u=this.u,v=this.v;
      const rhs=new Float32Array(N*N);
      p.fill(0);let maxDiv=0;
      for(let j=0;j<N;j++)for(let i=0;i<N;i++){
        const k=this.I(i,j);
        if(phi[k]>=0)continue;
        const div=(u[this.U(i+1,j)]-u[this.U(i,j)]+v[this.V(i,j+1)]-v[this.V(i,j)])/H;
        rhs[k]=div*H*H/DT;
        maxDiv=Math.max(maxDiv,Math.abs(div));
      }
      // Gauss-Seidel/SOR: p_air=0 and solid walls have zero normal flux.
      // The source uses a stronger preconditioned conjugate-gradient solver.
      const omega=1.48;
      for(let iter=0;iter<65;iter++){
        for(let j=0;j<N;j++)for(let i=0;i<N;i++){
          const k=this.I(i,j);if(phi[k]>=0)continue;
          let sum=0,diag=0;
          if(i>0){diag++;if(phi[k-1]<0)sum+=p[k-1];}
          if(i<N-1){diag++;if(phi[k+1]<0)sum+=p[k+1];}
          if(j>0){diag++;if(phi[k-N]<0)sum+=p[k-N];}
          if(j<N-1){diag++;if(phi[k+N]<0)sum+=p[k+N];}
          const desired=(sum-rhs[k])/Math.max(1,diag);
          p[k]+=omega*(desired-p[k]);
        }
      }
      const coeff=DT/H;
      for(let j=0;j<N;j++)for(let i=1;i<N;i++){
        const a=this.I(i-1,j),b=this.I(i,j),k=this.U(i,j);
        if(phi[a]<0||phi[b]<0)u[k]-=coeff*((phi[b]<0?p[b]:0)-(phi[a]<0?p[a]:0));
        else u[k]=0;
        u[k]=clip(u[k],-9,9);
      }
      for(let j=1;j<N;j++)for(let i=0;i<N;i++){
        const a=this.I(i,j-1),b=this.I(i,j),k=this.V(i,j);
        if(phi[a]<0||phi[b]<0)v[k]-=coeff*((phi[b]<0?p[b]:0)-(phi[a]<0?p[a]:0));
        else v[k]=0;
        v[k]=clip(v[k],-9,9);
      }
      for(let j=0;j<N;j++){u[this.U(0,j)]=0;u[this.U(N,j)]=0;}
      for(let i=0;i<N;i++){v[this.V(i,0)]=0;v[this.V(i,N)]=0;}
      // Report the achieved divergence, rather than just the pre-projection value.
      let measure=0,count=0;
      for(let j=1;j<N-1;j++)for(let i=1;i<N-1;i++){
        const k=this.I(i,j);
        if(phi[k]<-2*H){
          const d=(u[this.U(i+1,j)]-u[this.U(i,j)]+v[this.V(i,j+1)]-v[this.V(i,j)])/H;
          measure+=d*d;count++;
        }
      }
      this.divergence=Math.sqrt(measure/Math.max(count,1));
    }
    advectLevelSet(){
      for(let j=0;j<N;j++)for(let i=0;i<N;i++){
        const x=(i+.5)*H,y=(j+.5)*H,a=this.velocity(x,y);
        const b=this.velocity(x-.5*DT*a[0],y-.5*DT*a[1]);
        this.nextPhi[this.I(i,j)]=this.phiAt(x-DT*b[0],y-DT*b[1]);
      }
      const swap=this.phi;this.phi=this.nextPhi;this.nextPhi=swap;
    }
    correctArea(){
      if(!this.volumeCorrect)return;
      const target=this.initialArea,got=this.area(),difference=target-got;
      if(Math.abs(difference)<.0004)return;
      // A bounded, signed-distance shift offsets semi-Lagrangian mass loss.
      // This differs from the PI divergence controller in the C++ source.
      const phi=this.phi,epsilon=H*.8;
      let lo=-H*.6,hi=H*.6;
      for(let b=0;b<12;b++){
        const shift=(lo+hi)*.5;
        let sum=0;
        for(let k=0;k<phi.length;k++)sum+=clamp01(.5-(phi[k]+shift)/(2*epsilon));
        const volume=sum/(N*N);
        if(volume>target)lo=shift;
        else hi=shift;
      }
      const change=(lo+hi)*.5;
      for(let k=0;k<phi.length;k++)phi[k]+=change;
    }
    advance(){
      this.advectVelocity();
      this.applyForces();
      this.project();
      this.advectLevelSet();
      this.steps++;this.age+=DT;
      if(this.steps%6===0)this.reinitialize();
      // Apply the bounded shift each integration step; correcting only after
      // redistancing loses excessive area for the experimental cavity case.
      this.correctArea();
    }
  }

  let mode='dam',solver=new Solver(mode),playing=!reduced.matches,visible=true;
  let raf=0,last=0,accumulator=0;
  function updateControls(){
    solver.gravity=Number(gravity?.value||9.8);
    solver.sigma=Number(sigma?.value||0);
    solver.volumeCorrect=!!correction?.checked;
    if(gravityOut)gravityOut.textContent=solver.gravity.toFixed(1);
    if(sigmaOut)sigmaOut.textContent=solver.sigma.toFixed(3);
  }
  function format(a){return Number.isFinite(a)?a.toFixed(3):'—';}
  function redraw(){
    const w=canvas.width/(window.devicePixelRatio||1),h=canvas.height/(window.devicePixelRatio||1);
    // Browser resizing may cap DPR, so use the stored transform dimensions.
    const px=canvas.clientWidth||700,py=canvas.clientHeight||430;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    const ctxWidth=canvas.width/dpr,ctxHeight=canvas.height/dpr;
    ctx.clearRect(0,0,ctxWidth,ctxHeight);
    ctx.fillStyle='#031321';ctx.fillRect(0,0,ctxWidth,ctxHeight);
    const side=Math.min(ctxHeight-36,ctxWidth-50);
    const ox=(ctxWidth-side)*.5,oy=(ctxHeight-side)*.5;
    const X=x=>ox+x*side,Y=y=>oy+(1-y)*side;
    ctx.save();
    const background=ctx.createLinearGradient(0,oy,0,oy+side);
    background.addColorStop(0,'#061d30');background.addColorStop(1,'#0b2840');
    ctx.fillStyle=background;ctx.fillRect(ox,oy,side,side);
    if(!grid||grid.checked){
      ctx.strokeStyle='rgba(158,213,231,.10)';ctx.lineWidth=.8;
      for(let g=0;g<=10;g++){
        ctx.beginPath();ctx.moveTo(X(g/10),Y(0));ctx.lineTo(X(g/10),Y(1));ctx.stroke();
        ctx.beginPath();ctx.moveTo(X(0),Y(g/10));ctx.lineTo(X(1),Y(g/10));ctx.stroke();
      }
    }
    // Evaluate the cell-centered level set at mesh vertices for marching squares.
    const field=new Float32Array((N+1)*(N+1));
    for(let j=0;j<=N;j++)for(let i=0;i<=N;i++)
      field[i+j*(N+1)]=solver.phiAt(i/N,j/N);
    const watery=ctx.createLinearGradient(0,Y(1),0,Y(0));
    watery.addColorStop(0,'#44afcb');watery.addColorStop(1,'#0b507b');
    ctx.fillStyle=watery;
    const contours=[];
    function segment(a,b,f,g){
      const t=Math.abs(f-g)<1e-8?.5:clip(f/(f-g),0,1);
      return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];
    }
    ctx.beginPath();
    for(let j=0;j<N;j++)for(let i=0;i<N;i++){
      const k=i+j*(N+1);
      const ph=[field[k],field[k+1],field[k+N+2],field[k+N+1]];
      const pts=[[i/N,j/N],[(i+1)/N,j/N],[(i+1)/N,(j+1)/N],[i/N,(j+1)/N]];
      const poly=[],crossings=[];
      for(let c=0;c<4;c++){
        const d=(c+1)%4;
        if(ph[c]<0)poly.push(pts[c]);
        if((ph[c]<0)!==(ph[d]<0)){
          const pp=segment(pts[c],pts[d],ph[c],ph[d]);
          poly.push(pp);crossings.push(pp);
        }
      }
      if(poly.length>=3){
        ctx.moveTo(X(poly[0][0]),Y(poly[0][1]));
        for(let t=1;t<poly.length;t++)ctx.lineTo(X(poly[t][0]),Y(poly[t][1]));
        ctx.closePath();
      }
      if(crossings.length>=2)for(let t=0;t+1<crossings.length;t+=2)
        contours.push([crossings[t],crossings[t+1]]);
    }
    ctx.fill();
    ctx.strokeStyle='#99e6ef';ctx.lineWidth=2.3;ctx.lineJoin='round';
    ctx.beginPath();
    for(const [a,b] of contours){ctx.moveTo(X(a[0]),Y(a[1]));ctx.lineTo(X(b[0]),Y(b[1]));}
    ctx.stroke();
    if(vectors?.checked){
      ctx.strokeStyle='rgba(227,248,250,.60)';ctx.fillStyle='rgba(227,248,250,.70)';ctx.lineWidth=1;
      for(let y=1;y<12;y++)for(let x=1;x<12;x++){
        const xx=x/12,yy=y/12;
        if(solver.phiAt(xx,yy)>=0)continue;
        const v=solver.velocity(xx,yy),mag=Math.hypot(v[0],v[1]);
        if(mag<.03)continue;
        const len=Math.min(.034,.014*mag),dx=v[0]/mag*len,dy=v[1]/mag*len;
        const sx=X(xx),sy=Y(yy),ex=X(xx+dx),ey=Y(yy+dy);
        const ang=Math.atan2(ey-sy,ex-sx);
        ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ex,ey);ctx.stroke();
        ctx.beginPath();ctx.moveTo(ex,ey);
        ctx.lineTo(ex-4*Math.cos(ang-.55),ey-4*Math.sin(ang-.55));
        ctx.lineTo(ex-4*Math.cos(ang+.55),ey-4*Math.sin(ang+.55));
        ctx.closePath();ctx.fill();
      }
    }
    ctx.strokeStyle='rgba(160,213,230,.65)';ctx.lineWidth=1.4;ctx.strokeRect(ox,oy,side,side);
    ctx.restore();
    if(timeOut)timeOut.textContent=solver.age.toFixed(2);
    if(areaOut)areaOut.textContent=(100*solver.area()/Math.max(solver.initialArea,1e-8)).toFixed(1)+'%';
    if(divOut)divOut.textContent=solver.divergence.toExponential(1);
    if(status)status.textContent=scenarios[mode].title+' · '+solver.steps+' solver steps';
  }
  function resize(){
    const rect=canvas.getBoundingClientRect();
    const width=Math.max(240,Math.round(rect.width||650));
    const height=Math.round(width*.69);
    const dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    redraw();
  }
  function pause(){
    playing=false;
    if(run){run.setAttribute('aria-pressed','false');run.textContent='Play ▶';}
  }
  function resume(){
    playing=true;
    if(run){run.setAttribute('aria-pressed','true');run.textContent='Pause Ⅱ';}
    last=0;sync();
  }
  function sync(){
    if(raf){cancelAnimationFrame(raf);raf=0;}
    if(playing&&!document.hidden&&visible)raf=requestAnimationFrame(frame);
  }
  function frame(now){
    raf=0;if(!playing||!visible||document.hidden)return;
    const elapsed=last?Math.min(.06,(now-last)/1000):0;last=now;
    const speedValue=Number(speed?.value||1);
    accumulator+=elapsed*speedValue;
    let worked=0;
    while(accumulator>=DT&&worked<4){solver.advance();accumulator-=DT;worked++;}
    if(worked){redraw();if(solver.age>3.0){pause();}}
    raf=playing?requestAnimationFrame(frame):0;
  }
  function setMode(next){
    if(!(next in scenarios))return;
    mode=next;updateControls();solver=new Solver(mode);updateControls();
    if(description)description.textContent=scenarios[mode].description;
    tabs.forEach(b=>{
      const selected=b.dataset.levelsetCase===next;
      b.classList.toggle('active',selected);
      b.setAttribute('aria-pressed',String(selected));
    });
    accumulator=0;last=0;redraw();sync();
  }
  if(run)run.addEventListener('click',()=>{if(playing){pause();sync();}else resume();});
  if(reset)reset.addEventListener('click',()=>{solver.initialize(mode);accumulator=0;last=0;redraw();});
  if(stepButton)stepButton.addEventListener('click',()=>{pause();sync();updateControls();for(let i=0;i<4;i++)solver.advance();redraw();});
  tabs.forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.levelsetCase)));
  for(const input of [gravity,sigma,correction])if(input)input.addEventListener('input',()=>{updateControls();redraw();});
  for(const input of [grid,vectors])if(input)input.addEventListener('change',redraw);
  if('IntersectionObserver' in window)
    new IntersectionObserver(entries=>{visible=!!entries[0]&&entries[0].isIntersecting;last=0;sync();},{rootMargin:'100px 0px',threshold:0}).observe(canvas);
  window.addEventListener('resize',resize,{passive:true});
  document.addEventListener('visibilitychange',()=>{last=0;sync();});
  if(reduced.addEventListener)reduced.addEventListener('change',()=>{if(reduced.matches){pause();sync();}});
  if(reduced.matches)pause();
  if(description)description.textContent=scenarios[mode].description;
  updateControls();resize();sync();
})();
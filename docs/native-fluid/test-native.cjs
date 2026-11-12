/* CI smoke test for the complete REAL_FLUID C++ WebAssembly physics engine.
 * Tests only the two website-enabled original cases: dam and drop. */
'use strict';
const path=require('node:path');
const fs=require('node:fs');
const createFluidModule=require('./fluid-core.js');
async function main(){
  const module=await createFluidModule({
    locateFile:p=>path.join(__dirname,p),
    print:()=>{}, printErr:()=>{}
  });
  module._fluid_init(64);
  if(module._fluid_grid()!==64)throw Error('Grid size mismatch');
  module._fluid_set_physics(9.8,0.07);
  for(let mode=0;mode<2;mode++){
    module._fluid_choose_case(mode);
    const initial=module._fluid_volume();
    if(!Number.isFinite(initial) || initial<=0)throw Error('Initial volume invalid: '+mode+' / '+initial);
    // Exercise the native solver beyond its first timestep, not only the IC.
    for(let cycle=0;cycle<20;cycle++)module._fluid_run_steps(3);
    const volume=module._fluid_volume();
    const residual=module._fluid_pressure_residual();
    const ptr=module._fluid_phi_ptr()>>>2;
    const u=module._fluid_u_ptr()>>>2,v=module._fluid_v_ptr()>>>2;
    const heap=module.HEAPF32;
    if(!heap||ptr+64*64>heap.length||u+4096>heap.length||v+4096>heap.length)
      throw Error('Out-of-range native heap view');
    for(let i=0;i<4096;i++){
      if(!Number.isFinite(heap[ptr+i])||!Number.isFinite(heap[u+i])||!Number.isFinite(heap[v+i]))
        throw Error('Nonfinite field at case '+mode+' index '+i);
    }
    if(!Number.isFinite(volume)||!Number.isFinite(residual))
      throw Error('Nonfinite numerical result '+mode+': '+volume+'/'+residual);
    if(volume<=0||volume>3*initial)
      throw Error('Unphysical volume in short test '+mode+': '+volume+'/'+initial);
    console.log(JSON.stringify({case:mode,initialVolume:initial,volume,pressureResidual:residual,time:module._fluid_time()}));
  }
  // Start a fresh instance at the native desktop solver's default 128x128.
  // The original C++ arrays do not support resizing in place.
  const large=await createFluidModule({locateFile:p=>path.join(__dirname,p),print:()=>{},printErr:()=>{}});
  large._fluid_init(128);
  large._fluid_set_physics(9.8,0.07);
  large._fluid_run_steps(2);
  const largeArea=large._fluid_volume();
  if(large._fluid_grid()!==128 || !Number.isFinite(largeArea)||largeArea<=0)
    throw Error('128² native-grid smoke test failed');
  console.log(JSON.stringify({nativeGrid:128,volume:largeArea,residual:large._fluid_pressure_residual()}));
  if(!fs.existsSync(path.join(__dirname,'fluid-core.wasm')))throw Error('WASM artifact missing');
  console.log('Both dam-break and droplet-impact C++ / WASM initialization and time-step checks passed.');
}
main().catch(error=>{console.error(error);process.exit(1);});

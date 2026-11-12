/* Regression check for the screenshot-reported intro/video collision.
 * Serve the /docs folder at http://127.0.0.1:8765 before running:
 *   npm install --no-save playwright-core
 *   node docs/tests/check-hero-layout.cjs
 */
'use strict';
const assert=require('node:assert/strict');
const {chromium}=require('playwright-core');
const fs=require('node:fs');
const path=require('node:path');
const executable=['/usr/bin/google-chrome','/usr/bin/google-chrome-stable',
 '/usr/bin/chromium','/usr/bin/chromium-browser'].find(x=>fs.existsSync(x));
if(!executable)throw Error('No Chrome / Chromium executable found on runner');
async function main(){
 const browser=await chromium.launch({headless:true,executablePath:executable,args:['--no-sandbox','--disable-dev-shm-usage']});
 const report=[];
 try{
  for(const width of [1536,1366,1280,1150,1024,768,600,390]){
   const page=await browser.newPage({viewport:{width,height:1000},deviceScaleFactor:1});
   // Network-independent layout test: the remote movie cannot affect geometry.
   await page.route(/https?:\/\/(?!127\.0\.0\.1:8765)/,route=>route.abort());
   await page.goto('http://127.0.0.1:8765/',{waitUntil:'domcontentloaded'});
   const data=await page.evaluate(()=>{
    const box=(sel)=>{const el=document.querySelector(sel);if(!el)throw Error('Missing '+sel);
      const r=el.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};
    const intro=box('.cosmos-hero-inner .cosmos-copy');
    const film=box('.cosmos-hero-inner .cosmos-stage');
    const cards=[...document.querySelectorAll('.cosmos-hero-inner .hero-credential')].map(e=>{
      const r=e.getBoundingClientRect();return {x:r.x,right:r.right,y:r.y,bottom:r.bottom};
    });
    const controls=box('.cosmos-media-column .cosmo-simulator-controls');
    return {intro,film,cards,controls,stagePosition:getComputedStyle(document.querySelector('.cosmos-stage')).position,
      filmObjectFit:getComputedStyle(document.querySelector('.cosmos-film')).objectFit,
      viewport:innerWidth,scrollWidth:document.documentElement.scrollWidth,
      overflows:[...document.querySelectorAll('body *')].filter(e=>{
        const r=e.getBoundingClientRect(),cs=getComputedStyle(e);
        return r.width>0&&r.right>innerWidth+3&&cs.position!=='fixed';
      }).slice(0,12).map(e=>({tag:e.tagName.toLowerCase(),cls:String(e.className).slice(0,100),
        right:Math.round(e.getBoundingClientRect().right),width:Math.round(e.getBoundingClientRect().width)}))};
   });
   const intersects=(a,b)=>a.x<b.right-1 && a.right>b.x+1 && a.y<b.bottom-1 && a.bottom>b.y+1;
   assert(!intersects(data.intro,data.film),'Intro overlaps cosmic movie at '+width+'px');
   assert(!intersects(data.intro,data.controls),'Intro overlaps movie controls at '+width+'px');
   for(const card of data.cards)assert(!intersects(card,data.film),'Qualification overlaps movie at '+width+'px');
   assert(data.film.x>=-1&&data.film.right<=width+2,'Movie extends out of viewport at '+width+'px');
   assert(data.film.height>=120,'Movie collapsed at '+width+'px');
   assert(data.stagePosition==='relative','Movie should be in normal flow');
   assert(data.filmObjectFit==='contain','Cosmic video should preserve full frame');
   if(data.scrollWidth>width+3)console.warn('OVERFLOW DIAGNOSTICS',width,JSON.stringify(data.overflows));
   assert(data.scrollWidth<=width+3,'Unexpected horizontal scroll at '+width+'px');
   if(width>=1151){
    assert(data.film.x>=data.intro.right+8,'Expected two nonintersecting columns at '+width+'px');
   }else{
    assert(data.film.y>=data.intro.bottom+8,'Expected film below introduction at '+width+'px');
   }
   await page.screenshot({path:path.join('/tmp','hero-'+width+'.png'),fullPage:false});
   report.push({width,arrangement:width>=1151?'side-by-side':'stacked',
     introRight:Math.round(data.intro.right),filmLeft:Math.round(data.film.x),
     filmTop:Math.round(data.film.y),introBottom:Math.round(data.intro.bottom),
     scrollWidth:data.scrollWidth});
   await page.close();
  }
 }finally{await browser.close();}
 console.log(JSON.stringify({success:true,viewports:report},null,2));
}
main().catch(err=>{console.error(err);process.exit(1)});

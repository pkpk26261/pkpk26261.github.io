'use strict';
const { chromium } = require('playwright');
class EngineRunner {
  constructor(origin, options = {}) { this.origin=origin; this.options=options; this.context=null; this.page=null; this.generation=0; this.timer=null; this.busy=false; }
  async launch() {
    if(this.options.browserProvider){this.browser=await this.options.browserProvider();return;}
    this.browser = await chromium.launch({ headless:true, ...(this.options.channel ? {channel:this.options.channel} : {}),
      args:['--disable-background-timer-throttling','--disable-renderer-backgrounding','--autoplay-policy=no-user-gesture-required'] });
    this.browser.on('disconnected',()=>this.room?.failEngine('權威瀏覽器已停止，請重建比賽'));
  }
  async load(room, checkpoint) {
    const generation=++this.generation; this.room=room; clearInterval(this.timer);
    await this.context?.close();
    if(generation!==this.generation)return;
    if(!this.browser?.isConnected())await this.launch();
    this.context=await this.browser.newContext({ignoreHTTPSErrors:true,viewport:{width:960,height:540}});
    if(generation!==this.generation){await this.context.close();return;}
    const origin=this.origin;
    await this.context.route('**/*',route=>new URL(route.request().url()).origin===origin?route.continue():route.abort());
    this.page=await this.context.newPage();
    await this.page.exposeFunction('lanSnapshot',snapshot=>{if(generation===this.generation)room.receiveSnapshot(snapshot);});
    this.page.on('pageerror',err=>{if(generation!==this.generation)return;console.error('ENGINE:',err.message);room.failEngine('比賽引擎發生錯誤，請復原引擎或重建比賽');});
    this.page.on('crash',()=>{if(generation===this.generation)room.failEngine('權威運算程序中斷，請復原引擎或重建比賽');});
    const params=new URLSearchParams({lan:'authority',difficulty:room.difficulty,lanRevision:String(room.engineRevision)});
    for(const team of ['red','blue']){params.set(team+'Humans',room.roster[team].humans);params.set(team+'AI',room.roster[team].ai);}
    await this.page.goto(`${origin}/apps/drone-soccer/games/radio/drone_soccer_competition_add_ai_realgame.html?${params}`,{waitUntil:'load',timeout:30000});
    if(generation!==this.generation)return;
    await this.page.waitForFunction(()=>window.DroneLANEngine?.active);
    if(checkpoint)await this.page.evaluate(value=>window.DroneLANEngine.restore(value),{checkpoint,command:room.command});
    this.busy=false;
    const pump=async()=>{
      if(this.busy||generation!==this.generation)return;
      this.busy=true;
      try{await this.page.evaluate(value=>window.DroneLANEngine.apply(value),room.controls());}
      catch(err){if(generation===this.generation&&room.engineReady)room.failEngine('權威運算連線中斷，請重建比賽');}
      finally{if(generation===this.generation)this.busy=false;}
    };
    await pump(); this.timer=setInterval(pump,30);
  }
  async stop(){++this.generation;clearInterval(this.timer);this.room=null;await this.context?.close();this.context=null;this.page=null;}
  async close(){await this.stop();if(!this.options.browserProvider)await this.browser?.close();}
}
class EnginePool {
  constructor(origin,options={}){this.origin=origin;this.options=options;this.runners=new Map();this.queues=new Map();this.closing=false;}
  async launch(){
    if(this.browser?.isConnected())return this.browser;
    if(this.launching)return this.launching;
    this.launching=chromium.launch({headless:true,...(this.options.channel?{channel:this.options.channel}:{}),args:['--disable-background-timer-throttling','--disable-renderer-backgrounding','--autoplay-policy=no-user-gesture-required']}).then(browser=>{
      this.browser=browser;browser.on('disconnected',()=>{if(!this.closing)for(const runner of this.runners.values())if(runner.room&&!['ended','closed'].includes(runner.room.status))runner.room.failEngine('賽場運算服務中斷，請由主持人復原');});return browser;
    }).finally(()=>this.launching=null);return this.launching;
  }
  load(room,checkpoint){
    let runner=this.runners.get(room.code);
    if(!runner){runner=new EngineRunner(this.origin,{browserProvider:()=>this.launch()});this.runners.set(room.code,runner);}
    const task=(this.queues.get(room.code)||Promise.resolve()).catch(()=>{}).then(async()=>{
      if(this.closing||this.runners.get(room.code)!==runner||['closed','ended'].includes(room.status))return;
      await runner.load(room,checkpoint);
    }).catch(error=>{if(!this.closing&&!['closed','ended'].includes(room.status))room.failEngine('賽場引擎啟動失敗：'+error.message);});
    this.queues.set(room.code,task);return task;
  }
  async stop(code){const runner=this.runners.get(code);this.runners.delete(code);this.queues.delete(code);await runner?.stop();}
  async close(){this.closing=true;await Promise.all([...this.runners.keys()].map(code=>this.stop(code)));await this.browser?.close();}
}
module.exports={EngineRunner,EnginePool};

'use strict';
// Isolated HTTP backend + real Chromium. HTTPS is terminated by the deployment
// proxy; these checks deliberately do not claim a public-cloud deployment.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),http=require('node:http'),net=require('node:net'),{randomBytes}=require('node:crypto');
const bundledBrowsers=path.resolve(__dirname,'../../runtime/ms-playwright');
if(!process.env.PLAYWRIGHT_BROWSERS_PATH&&fs.existsSync(bundledBrowsers))process.env.PLAYWRIGHT_BROWSERS_PATH=bundledBrowsers;
const {WebSocket}=require('ws'),{start}=require('../server.cjs'),{ServiceStore}=require('../service-store.cjs'),{EnginePool}=require('../engine-runner.cjs');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(predicate,label,timeout=45000){const deadline=Date.now()+timeout;while(Date.now()<deadline){if(await predicate())return;await sleep(30);}throw Error('Timeout: '+label);}
async function freePort(){const server=net.createServer();await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});const port=server.address().port;await new Promise(resolve=>server.close(resolve));return port;}
function request(origin,url,{method='GET',data,requestOrigin='https://flight-club.example',cookie,headers={}}={}){return new Promise((resolve,reject)=>{
 const body=data===undefined?null:JSON.stringify(data),req=http.request(origin+url,{method,headers:{...(requestOrigin===null?{}:{Origin:requestOrigin}),...(body?{'Content-Type':'application/json','Content-Length':Buffer.byteLength(body)}:{}),...(cookie?{Cookie:cookie}:{}),...headers}},res=>{let text='';res.setEncoding('utf8');res.on('data',chunk=>text+=chunk);res.on('end',()=>resolve({status:res.statusCode,body:text,headers:res.headers,json:()=>JSON.parse(text)}));});
 req.on('error',reject);req.setTimeout(10000,()=>req.destroy(Error('HTTP request timeout')));if(body)req.write(body);req.end();
});}
async function client(origin,clients,{requestOrigin='https://flight-club.example',cookie}={}){
 const result={messages:[]};result.ws=new WebSocket(origin.replace('http:','ws:')+'/socket',{origin:requestOrigin,...(cookie?{headers:{Cookie:cookie}}:{})});clients.push(result.ws);
 result.ws.on('message',raw=>result.messages.push(JSON.parse(raw)));result.ws.on('error',()=>{});
 await new Promise((resolve,reject)=>{result.ws.once('open',resolve);result.ws.once('error',reject);});result.send=message=>result.ws.send(JSON.stringify(message));return result;
}
function rejectedSocket(origin,requestOrigin,headers={}){return new Promise((resolve,reject)=>{
 const ws=new WebSocket(origin.replace('http:','ws:')+'/socket',{...(requestOrigin===null?{}:{origin:requestOrigin}),headers}),timer=setTimeout(()=>{ws.terminate();reject(Error('Rejected socket did not close'));},5000);
 ws.once('open',()=>{clearTimeout(timer);ws.terminate();reject(Error('Unexpected Origin was accepted'));});ws.once('error',()=>{clearTimeout(timer);resolve();});ws.once('close',()=>clearTimeout(timer));
});}
(async()=>{
 const envKeys=['PORT','PUBLIC_ORIGIN','RENDER_EXTERNAL_URL','ADMIN_TOKEN','DATA_DIR','MAX_ARENAS'],previous=Object.fromEntries(envKeys.map(key=>[key,process.env[key]]));
 const testRoot=fs.mkdtempSync(path.join(os.tmpdir(),'flight-club-cloud-')),clients=[],logs=[],originalLog=console.log,originalLaunch=EnginePool.prototype.launch;
 const publicOrigin='https://flight-club.example',secret=randomBytes(32).toString('base64url'),report={kind:'Isolated cloud HTTP backend, real bundled Chromium, exact public Origin/authentication, persistence; no production host or public TLS deployment',checks:[]};let service;
 console.log=(...values)=>logs.push(values.join(' '));
 try{
  for(const key of envKeys)delete process.env[key];
  process.env.PUBLIC_ORIGIN=publicOrigin;process.env.DATA_DIR=path.join(testRoot,'state');process.env.PORT=String(await freePort());
  await assert.rejects(()=>start({cloud:true}),/ADMIN_TOKEN/);
  process.env.ADMIN_TOKEN='short';await assert.rejects(()=>start({cloud:true}),/ADMIN_TOKEN/);
  process.env.ADMIN_TOKEN=secret;
  delete process.env.PUBLIC_ORIGIN;await assert.rejects(()=>start({cloud:true}),/HTTPS/);
  for(const origin of ['http://flight-club.example','https://user:secret@flight-club.example','https://flight-club.example/play']){process.env.PUBLIC_ORIGIN=origin;await assert.rejects(()=>start({cloud:true}),/HTTPS/);}
  process.env.PUBLIC_ORIGIN=publicOrigin;await assert.rejects(()=>start({cloud:true,online:true}),/不可同時/);await assert.rejects(()=>start({cloud:true,'setup-port':19001}),/HTTP/);
  process.env.MAX_ARENAS='17';await assert.rejects(()=>start({cloud:true}),/賽場上限/);delete process.env.MAX_ARENAS;
  const selectedPort=process.env.PORT;process.env.PORT='1023';await assert.rejects(()=>start({cloud:true}),/連接埠/);process.env.PORT=selectedPort;
  report.checks.push('Cloud mode requires a strong configured secret and clean HTTPS origin, rejects conflicting legacy setup, and never reads a TLS certificate');
  EnginePool.prototype.launch=async()=>{throw Error('Intentional isolated browser launch failure');};
  await assert.rejects(()=>start({cloud:true,persist:false}),/Intentional isolated browser launch failure/);EnginePool.prototype.launch=originalLaunch;
  await assert.rejects(()=>request('http://127.0.0.1:'+process.env.PORT,'/api/health'));
  report.checks.push('A failed browser startup closes its HTTP listener instead of leaving a falsely healthy service');
  // Hold the real browser launch to observe startup readiness through HTTP.
  let releaseLaunch,launchEntered=false;const gate=new Promise(resolve=>{releaseLaunch=resolve;});
  EnginePool.prototype.launch=async function(){launchEntered=true;await gate;return originalLaunch.call(this);};
  const starting=start({cloud:true,'cert-dir':path.join(testRoot,'missing-certificates')});const internalOrigin='http://127.0.0.1:'+process.env.PORT;
  let observationFailure;
  try{await until(()=>launchEntered,'listener before Chromium launch',5000);const before=await request(internalOrigin,'/api/health');assert.equal(before.status,503);assert.equal(before.json().status,'starting');}catch(error){observationFailure=error;}finally{releaseLaunch();}
  service=await starting;EnginePool.prototype.launch=originalLaunch;
  if(observationFailure)throw observationFailure;
  assert.equal(service.server.address().address,'0.0.0.0');assert.equal(service.internalOrigin,internalOrigin);assert.equal(service.engines.origin,internalOrigin);assert.equal(service.origin,publicOrigin);assert.equal(service.arenas.length,2);assert.equal(service.directory().maxArenas,2);assert.equal(service.adminToken,secret);
  const health=await request(internalOrigin,'/api/health');assert.equal(health.status,200);assert.equal(health.json().status,'ok');
  const config=(await request(internalOrigin,'/api/config')).json();assert.equal(config.origin,publicOrigin);assert.equal(config.accessMode,'online');assert.equal(config.publicReady,true);assert.equal(config.directJoin,true);assert.equal(config.certificateTrustRequired,false);assert.equal(config.setupURL,null);assert.equal(config.joinURL,publicOrigin+'/play');assert(!('addresses' in config));assert(!JSON.stringify(config).includes('127.0.0.1'));
  for(const route of ['/','/admin','/play','/watch','/apps/drone-soccer/'])assert.equal((await request(internalOrigin,route)).status,200,route);
  const help=await request(internalOrigin,'/apps/drone-soccer/lan/setup.html');assert.equal(help.status,200);assert(help.body.includes('線上多人指南'));assert(!help.body.includes('FlightClub.exe'));
  assert.equal((await request(internalOrigin,'/ca.crt')).status,404);assert.equal((await request(internalOrigin,'/api/public-origin',{method:'POST',data:{origin:'https://attacker.example'}})).status,404);
  assert(!logs.join('\n').includes(secret));assert(!logs.join('\n').includes('#admin='));
  report.checks.push('HTTP binds all container interfaces, authority stays loopback, public config remains HTTPS-only without CA/setup/private addresses, and health is 503 until Chromium starts; startup logs omit the secret');
  for(const origin of ['https://evil.example',publicOrigin+'.evil.example','null',null]){
   const headers={'X-Forwarded-Host':'flight-club.example','X-Forwarded-Proto':'https'};
   assert.equal((await request(internalOrigin,'/api/login',{method:'POST',requestOrigin:origin,data:{key:secret},headers})).status,403);await rejectedSocket(internalOrigin,origin,headers);
  }
  const anonymous=await client(internalOrigin,clients);anonymous.send({type:'create',roster:{red:{humans:0,ai:1},blue:{humans:0,ai:1}}});await until(()=>anonymous.messages.some(m=>m.type==='error'),'anonymous room creation denied');
  const login=await request(internalOrigin,'/api/login',{method:'POST',data:{key:secret}});assert.equal(login.status,200);const setCookie=login.headers['set-cookie'][0],cookie=setCookie.split(';')[0];for(const attribute of ['HttpOnly','Secure','SameSite=Strict','Path=/'])assert(setCookie.includes(attribute));
  const host=await client(internalOrigin,clients,{cookie});host.send({type:'create',roster:{red:{humans:0,ai:1},blue:{humans:0,ai:1}},name:'Cloud integration room'});
  await until(()=>host.messages.some(m=>m.type==='created'),'authenticated room creation');const created=host.messages.find(m=>m.type==='created'),room=service.roomMap.get(created.room.code);
  await until(()=>room.engineReady&&room.snapshot,'real cloud authority snapshot');assert(service.engines.runners.get(room.code).page.url().startsWith(internalOrigin+'/apps/drone-soccer/games/radio/'));
  const qr=await request(internalOrigin,'/api/qr?code='+room.code);assert.equal(qr.status,200);assert(qr.headers['content-type'].includes('image/svg+xml'));
  host.send({type:'start'});await until(()=>room.status==='running'&&room.snapshot?.match?.phase==='play','real cloud match start');const savedCode=room.code;
  report.checks.push('Exact public Origin and Secure cookie login permit cloud WSS management; anonymous creation and forged forwarding Origins are rejected; real Chromium loads loopback game and produces a running match');
  assert.equal((await request(internalOrigin,'/api/logout',{method:'POST',cookie})).status,200);await until(()=>host.messages.some(m=>m.type==='signedOut'),'live host logout');
  for(const ws of clients)ws.terminate();clients.length=0;await service.close();service=null;
  assert.equal(JSON.parse(fs.readFileSync(path.join(process.env.DATA_DIR,'state.json'),'utf8')).rooms[0].code,savedCode);
  // A deployment may use Render's assigned HTTPS hostname without PUBLIC_ORIGIN.
  delete process.env.PUBLIC_ORIGIN;process.env.RENDER_EXTERNAL_URL=publicOrigin;process.env.PORT=String(await freePort());process.env.MAX_ARENAS='2';
  service=await start({cloud:true});assert.equal(service.origin,publicOrigin);assert.equal(service.adminToken,secret);assert.equal(service.roomMap.get(savedCode).status,'paused');await until(()=>service.roomMap.get(savedCode).engineReady,'restored authority');
  const replacement=randomBytes(32).toString('base64url'),rotated=new ServiceStore({dataDir:path.join(testRoot,'rotation'),adminToken:secret,maxArenas:2});rotated.write({...rotated.state,siteName:'Preserved cloud name'});const reopened=new ServiceStore({dataDir:path.join(testRoot,'rotation'),adminToken:replacement,maxArenas:2});assert.equal(reopened.state.adminToken,replacement);assert.equal(reopened.state.siteName,'Preserved cloud name');
  report.checks.push('Secure cookie logout revokes an active host, shutdown persists room state, Render HTTPS fallback restores the room paused, and deployment-secret rotation preserves existing settings');
  report.result='PASS';
 }catch(error){report.result='FAIL';report.failure=error.stack;process.exitCode=1;}
 finally{
  EnginePool.prototype.launch=originalLaunch;for(const ws of clients)ws.terminate();await service?.close();console.log=originalLog;for(const key of envKeys)if(previous[key]===undefined)delete process.env[key];else process.env[key]=previous[key];
  const resolved=path.resolve(testRoot),temporary=path.resolve(os.tmpdir())+path.sep;if(!resolved.startsWith(temporary)||!path.basename(resolved).startsWith('flight-club-cloud-'))throw Error('Refusing unexpected temporary cleanup path');fs.rmSync(resolved,{recursive:true,force:true});
  console.log(JSON.stringify(report,null,2));
 }
})();

'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{randomBytes}=require('node:crypto');
class ServiceStore {
  constructor(options={}){
    if(options.adminToken!==undefined&&(typeof options.adminToken!=='string'||options.adminToken.length<32||options.adminToken.length>512||/[\x00-\x20\x7f]/.test(options.adminToken)))throw Error('ADMIN_TOKEN 需為 32～512 字元，且不可包含空白或控制字元');
    this.file=options.persist===false?null:path.join(path.resolve(options.dataDir||path.join(os.homedir(),'.webduino-lan/service')),'state.json');
    this.state={version:1,adminToken:options.adminToken||randomBytes(32).toString('base64url'),siteName:options.siteName||'Flight Club 本機賽事服務',
      arenas:Array.from({length:Math.min(4,options.maxArenas||8)},(_,i)=>({id:'arena-'+(i+1),name:String.fromCharCode(65+i)+' 賽場',note:'正式賽・獨立運算',enabled:true})),rooms:[],closedCodes:[]};
    if(this.file&&fs.existsSync(this.file)){
      const data=JSON.parse(fs.readFileSync(this.file,'utf8'));
      if(data.version!==1||typeof data.adminToken!=='string'||!Array.isArray(data.arenas)||!Array.isArray(data.rooms))throw Error('服務資料格式不正確；請保留 state.json 並從備份修復，未覆寫原資料');
      this.state=data;
    }
    // A configured deployment secret is authoritative after a deliberate
    // rotation; room capabilities and match state stay intact.
    if(options.adminToken!==undefined)this.state.adminToken=options.adminToken;
    this.write(this.state);
  }
  write(state){
    this.state=state;if(!this.file)return;
    fs.mkdirSync(path.dirname(this.file),{recursive:true});const temp=this.file+'.tmp';
    fs.writeFileSync(temp,JSON.stringify(state),{mode:0o600});
    if(fs.existsSync(this.file))fs.copyFileSync(this.file,this.file+'.bak');
    fs.renameSync(temp,this.file);
  }
}
module.exports={ServiceStore};

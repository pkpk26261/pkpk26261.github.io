/* Flexible, open rope net. Local deformation modes stand in for many small
   5 cm mesh cells; the 20 cm protective cage cannot fit through a cell.
   Parameters are game calibration, not measurements of a particular net. */
(() => {
  'use strict';
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  class Net {
    constructor(bounds,axis='y',sign=-1) {
      this.bounds = bounds;
      this.axis=axis;this.sign=sign;this.outward=-sign;
      this.uAxis=axis==='x'?'z':'x';this.vAxis=axis==='y'?'z':'y';
      this.base=bounds[axis][sign>0?0:1];this.height=this.base;
      this.modes = [];
      this.visuals = [];
      this.dirty = true;
      this.parameters = Object.freeze({meshPitch:0.05,mass:0.06,stiffness:70,stretch:1800,damping:4.5});
    }
    weight(mode,x,z) {
      const dx=x-mode.x,dz=z-mode.z,r2=mode.radius**2,s=1-(dx*dx+dz*dz)/r2;
      return s>0 ? s*s : 0;
    }
    sample(x,z) {
      let displacement=0,velocity=0,gx=0,gz=0,inverseMassY=0;
      for(const m of this.modes) {
        const w=this.weight(m,x,z);if(!w)continue;
        const s=Math.sqrt(w),scale=-4*s/(m.radius*m.radius);
        displacement+=m.displacement*w;velocity+=m.velocity*w;
        gx+=m.displacement*scale*(x-m.x);gz+=m.displacement*scale*(z-m.z);
        inverseMassY+=w*w/this.parameters.mass;
      }
      return {height:this.height+displacement,displacement,velocity,gx,gz,inverseMassY};
    }
    engage(x,z) {
      if(!this.modes.some(m=>this.weight(m,x,z)>0.4)) {
        const u=this.bounds[this.uAxis],v=this.bounds[this.vAxis];
        const room=Math.min(x-u[0],u[1]-x,z-v[0],v[1]-z);
        if(room>0.015)this.modes.push({x,z,radius:Math.min(0.45,room),displacement:0,velocity:0});
      }
      return this.sample(x,z);
    }
    advance(delta) {
      const dt=clamp(Number.isFinite(delta)?delta:0,0,0.1),steps=Math.max(1,Math.ceil(dt*960)),h=dt/steps;
      const p=this.parameters;
      for(let i=0;i<steps;i++)for(const m of this.modes) {
        const restoring=p.stiffness*m.displacement+p.stretch*m.displacement**3;
        m.velocity=(m.velocity-restoring/p.mass*h)*Math.exp(-p.damping/p.mass*h);
        m.displacement+=m.velocity*h;
      }
      this.modes=this.modes.filter(m=>Math.abs(m.displacement)>1e-5||Math.abs(m.velocity)>1e-4);
      this.dirty=true;
    }
    sampleAt(point) {return this.sample(point[this.uAxis],point[this.vAxis]);}
    limit(point,radius) {
      const s=this.sampleAt(point);
      return this.base+this.outward*s.displacement+this.sign*radius*Math.sqrt(1+s.gx*s.gx+s.gz*s.gz);
    }
    normal(point) {
      const s=this.sampleAt(point),n={x:0,y:0,z:0},size=Math.hypot(s.gx,1,s.gz);
      n[this.axis]=this.sign/size;n[this.uAxis]=s.gx/size;n[this.vAxis]=s.gz/size;return n;
    }
    sweep(start,move,radius) {
      if(!this.modes.length) {
        const limit=this.base+this.sign*radius;
        const before=(limit-start[this.axis])*this.sign,after=before-move[this.axis]*this.sign;
        if(before<=1e-10&&after>1e-10)
          return {time:Math.max(0,-before/(after-before)),normal:this.normal(start),net:this};
        return null;
      }
      const at=t=>({x:start.x+move.x*t,y:start.y+move.y*t,z:start.z+move.z*t});
      const gap=t=>{const p=at(t);return (this.limit(p,radius)-p[this.axis])*this.sign;};
      let previous=0;
      for(let i=1;i<=16;i++) {
        const t=i/16;
        if(gap(t)>1e-10&&gap(previous)<=1e-10) {
          let lo=previous,hi=t;
          for(let j=0;j<24;j++){const mid=(lo+hi)/2;if(gap(mid)>0)hi=mid;else lo=mid;}
          return {time:hi,normal:this.normal(at(hi)),net:this};
        }
        previous=t;
      }
      return null;
    }
    contact(actor,point) {
      const u=point[this.uAxis],v=point[this.vAxis],s=this.engage(u,v),T=window.THREE;
      const normal=new T.Vector3().copy(this.normal(point)),velocity=new T.Vector3();velocity[this.axis]=this.outward*s.velocity;
      return window.DroneFlightDynamics.materialContact(actor,normal,'rope',{
        velocity,inverseMass:s.inverseMassY,axis:this.axis,point,
        applyImpulse:impulse=>{
          for(const m of this.modes)m.velocity-=impulse[this.axis]*this.outward*this.weight(m,u,v)/this.parameters.mass;
        }
      });
    }
    resolve(actor,start,radius=0.1) {
      const move={x:actor.position.x-start.x,y:actor.position.y-start.y,z:actor.position.z-start.z};
      const hit=this.sweep(start,move,radius);
      if(hit) {
        actor.position.set(start.x+move.x*hit.time,start.y+move.y*hit.time,start.z+move.z*hit.time);
        this.contact(actor,actor.position);
      }
      const limit=this.limit(actor.position,radius);
      if((actor.position[this.axis]-limit)*this.sign<0) {actor.position[this.axis]=limit;this.contact(actor,actor.position);}
      actor.mesh?.position.copy(actor.position);
      this.updateVisuals();
    }
    attach(group) {
      const T=window.THREE;
      group.updateMatrixWorld(true);
      group.traverse(mesh=>{
        const p=mesh.geometry?.parameters;
        if(!mesh.isMesh||!p?.width||!p.height||!mesh.material?.map||!mesh.material.transparent)return;
        const center=mesh.getWorldPosition(new T.Vector3());
        const normal=new T.Vector3(0,0,1).transformDirection(mesh.matrixWorld);
        if(Math.abs(normal[this.axis])<0.99||Math.abs(center[this.axis]-this.base)>0.02)return;
        mesh.geometry=new T.PlaneGeometry(p.width,p.height,Math.ceil(p.width/0.125),Math.ceil(p.height/0.125));
        const positions=mesh.geometry.attributes.position,rest=positions.array.slice(),world=[];
        const inverse=mesh.matrixWorld.clone().invert();
        const outward=new T.Vector3();outward[this.axis]=this.outward;
        const up=outward.applyMatrix4(inverse).sub(new T.Vector3().applyMatrix4(inverse));
        for(let i=0;i<positions.count;i++)world.push(new T.Vector3().fromBufferAttribute(positions,i).applyMatrix4(mesh.matrixWorld));
        mesh.name=`flexible-rope-${this.axis}-${this.sign}`;mesh.frustumCulled=false;
        const beforeRender=mesh.onBeforeRender;
        mesh.onBeforeRender=(...args)=>{beforeRender?.apply(mesh,args);this.updateVisuals();};
        this.visuals.push({mesh,positions,rest,world,up});
      });
    }
    updateVisuals() {
      if(!this.dirty)return;
      for(const v of this.visuals) {
        for(let i=0;i<v.positions.count;i++) {
          const w=this.sampleAt(v.world[i]).displacement;
          v.positions.setXYZ(i,v.rest[i*3]+v.up.x*w,v.rest[i*3+1]+v.up.y*w,v.rest[i*3+2]+v.up.z*w);
        }
        v.positions.needsUpdate=true;
      }
      this.dirty=false;
    }
    energy() {
      const p=this.parameters;
      return this.modes.reduce((sum,m)=>sum+0.5*p.mass*m.velocity*m.velocity+0.5*p.stiffness*m.displacement*m.displacement+0.25*p.stretch*m.displacement**4,0);
    }
    reset() {this.modes=[];this.dirty=true;this.updateVisuals();}
    snapshot() {return {axis:this.axis,sign:this.sign,height:this.height,parameters:this.parameters,energy:this.energy(),modes:this.modes.map(m=>({...m})),visuals:this.visuals.length};}
  }
  window.DroneRopeNet=Object.freeze({create:(bounds,axis,sign)=>new Net(bounds,axis,sign)});
})();

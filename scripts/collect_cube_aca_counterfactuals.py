#!/usr/bin/env python3
"""Mine positive ACA hinges on a fixed Cube subset and execute them in MuJoCo."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import h5py
import hdf5plugin
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import build_jepa

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--source', type=Path, default=ROOT/'data/external/cube_single_expert_train.h5')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rho', type=float, default=.2)
    p.add_argument('--margin', type=float, default=0.)
    p.add_argument('--top-fraction', type=float, default=.1)
    p.add_argument('--max-executions', type=int, default=None,
                   help='Optional cap on selected transitions before MuJoCo execution.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--episode-indices', type=Path, required=True)
    p.add_argument('--source-indices', type=Path, default=None,
                   help='Optional exact source transition rows. With --top-fraction=1, '
                        'this executes ACA actions for every supplied row without hinge filtering.')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()

def load(run, ckpt, device):
    cfg = OmegaConf.load(run/'config.yaml'); model = build_jepa(cfg)
    data = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = {k[6:]: v for k,v in data['state_dict'].items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected: raise RuntimeError(f'model mismatch: {missing} {unexpected}')
    model.to(device).eval().requires_grad_(False); model.interpolate_pos_encoding = True
    return model, cfg

def valid(h, fs, episodes):
    ep = h['ep_idx'][:]; rows = np.arange(len(ep)-fs, dtype=np.int64)
    rows = rows[ep[:-fs] == ep[fs:]]
    keep = set(np.load(episodes).astype(np.int64).tolist())
    return rows[np.array([int(ep[i]) in keep for i in rows], dtype=bool)]

def main():
    a = args(); run = a.run_dir.expanduser().resolve(); source = a.source.expanduser().resolve()
    ckpt = (a.checkpoint or run/'checkpoints/last.ckpt').expanduser().resolve(); dev = torch.device(a.device)
    model, cfg = load(run, ckpt, dev); fs = int(cfg.data.dataset.frameskip)
    with h5py.File(source, 'r') as h:
        rows = valid(h, fs, a.episode_indices)
        if a.source_indices is not None:
            requested = np.asarray(np.load(a.source_indices), dtype=np.int64)
            if requested.ndim != 1 or len(requested) == 0 or len(np.unique(requested)) != len(requested):
                raise ValueError('--source-indices must be a non-empty 1-D array without duplicates')
            allowed = set(int(x) for x in rows.tolist())
            if any(int(x) not in allowed for x in requested):
                raise ValueError('--source-indices contains rows outside the requested episode pool')
            rows = np.sort(requested)
        raw_all = h['action'][:].astype(np.float32)
        raw_all = raw_all[np.isfinite(raw_all).all(1)]; mean=raw_all.mean(0); std=np.maximum(raw_all.std(0),1e-6); low=raw_all.min(0); high=raw_all.max(0)
        tm=np.tile(mean,fs); ts=np.tile(std,fs); lo=torch.as_tensor((np.tile(low,fs)-tm)/ts,device=dev).view(1,1,-1); hi=torch.as_tensor((np.tile(high,fs)-tm)/ts,device=dev).view(1,1,-1)
        positive=[]; hinges=[]; all_rows=[]; all_hinges=[]
        for b in range(0,len(rows),a.batch_size):
            r=rows[b:b+a.batch_size]; pix=np.stack([h['pixels'][r],h['pixels'][r+fs]],1); pix=torch.from_numpy(pix).permute(0,1,4,2,3).to(dev)
            raw=np.stack([h['action'][int(i):int(i)+fs] for i in r]).reshape(len(r),-1).astype(np.float32); act=torch.as_tensor((raw-tm)/ts,device=dev).unsqueeze(1)
            with torch.no_grad(): z=model.encode({'pixels':pix})['emb']; z0,z1=z[:,:1],z[:,1:2]
            ag=act.detach().clone().requires_grad_(True); ae=model.action_encoder(ag) if model.action_encoder is not None else ag
            e=(model.predict(z0,ae)-z1).float().square().mean(dim=(-2,-1)); (g,)=torch.autograd.grad(e.sum(),ag); mined=(act-a.rho*g/g.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(lo,hi)
            with torch.no_grad():
                am=model.action_encoder(mined) if model.action_encoder is not None else mined; em=(model.predict(z0,am)-z1).float().square().mean(dim=(-2,-1)); hg=a.margin+e-em
            mask=hg.detach().cpu().numpy()>0
            all_rows.append(r.copy()); all_hinges.append(hg.detach().cpu().numpy())
            if mask.any(): positive.append(r[mask]); hinges.append(hg.detach().cpu().numpy()[mask])
            if b==0 or b+len(r)==len(rows): print(f'mined {b+len(r):,}/{len(rows):,}',flush=True)



        if a.source_indices is not None and a.top_fraction >= 1.0:
            rows = np.concatenate(all_rows)
            hinges = np.concatenate(all_hinges)
            print(f'fixed source pool; selected={len(rows):,} (no hinge filtering)', flush=True)
        elif positive:
            positive=np.concatenate(positive); hinges=np.concatenate(hinges)
            n=max(1,int(np.ceil(a.top_fraction*len(positive))))
            order=np.argsort(hinges)[-n:][::-1]; rows=positive[order]; hinges=hinges[order]
            print(f'positive={len(positive):,}; selected={len(rows):,}',flush=True)
        else:



            rows_all=np.concatenate(all_rows); hinges_all=np.concatenate(all_hinges)
            n=max(1,int(np.ceil(a.top_fraction*len(rows_all))))
            order=np.argsort(hinges_all)[-n:][::-1]; rows=rows_all[order]; hinges=hinges_all[order]
            print(f'positive=0; fallback=top_{n:,}_by_hinge; selected={len(rows):,}',flush=True)
        if a.max_executions is not None:
            if a.max_executions <= 0:
                raise ValueError('--max-executions must be positive')
            rows, hinges = rows[:a.max_executions], hinges[:a.max_executions]
        print(f'executing {len(rows):,} selected transitions', flush=True)
        import gymnasium as gym
        import stable_worldmodel.envs
        env=gym.make('swm/OGBCube-v0',env_type='single',ob_type='states',multiview=False,width=224,height=224,visualize_info=False,terminate_at_goal=True); u=env.unwrapped
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if a.output.exists():
            with h5py.File(a.output,'r') as old:
                if 'action' in old and len(old['action']): raise FileExistsError(a.output)
            a.output.unlink()
        out=h5py.File(a.output,'w'); out.attrs['format']='cube_aca_counterfactual_v1'; out.attrs['atomic_action_dim']=5; out.attrs['frameskip']=fs; out.attrs['source_run_dir']=str(run); out.attrs['source_checkpoint']=str(ckpt)
        out.create_dataset('pixels',shape=(0,2,224,224,3),maxshape=(None,2,224,224,3),dtype=np.uint8,chunks=(1,2,224,224,3),compression='lzf'); out.create_dataset('action',shape=(0,25),maxshape=(None,25),dtype=np.float32); out.create_dataset('observation',shape=(0,2,28),maxshape=(None,2,28),dtype=np.float32); out.create_dataset('source_index',shape=(0,),maxshape=(None,),dtype=np.int64); out.create_dataset('hinge',shape=(0,),maxshape=(None,),dtype=np.float32)







        def take_sorted(ds, indices):
            indices = np.asarray(indices, dtype=np.int64)
            order = np.argsort(indices)
            values = np.asarray(ds[indices[order]])
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            return values[inverse]

        selected_pixels = take_sorted(h['pixels'], rows)
        selected_pixels_next = take_sorted(h['pixels'], rows + fs)
        selected_actions = [np.asarray(h['action'][int(row):int(row)+fs]) for row in rows]
        selected_obs = take_sorted(h['observation'], rows).astype(np.float32)
        selected_qpos = take_sorted(h['qpos'], rows)
        selected_qvel = take_sorted(h['qvel'], rows)
        selected_targets = take_sorted(h['privileged_target_block_pos'], rows)
        try:
            for count,(row,hg) in enumerate(zip(rows,hinges),1):
                j = count - 1
                env.reset(seed=int(count+a.seed)); u.set_target_pos(0,selected_targets[j]); u.set_state(selected_qpos[j],selected_qvel[j]); start=selected_pixels[j]; obs0=selected_obs[j]
                pix=np.stack([selected_pixels[j], selected_pixels_next[j]])
                t=torch.from_numpy(pix[None]).permute(0,1,4,2,3).to(dev); raw=selected_actions[j].reshape(1,-1).astype(np.float32); act=torch.as_tensor((raw-tm)/ts,device=dev).unsqueeze(1)
                with torch.no_grad(): z=model.encode({'pixels':t})['emb']
                ag=act.detach().clone().requires_grad_(True); ae=model.action_encoder(ag) if model.action_encoder is not None else ag; e=(model.predict(z[:,:1],ae)-z[:,1:2]).float().square().mean(); (g,)=torch.autograd.grad(e,ag); cf=(act-a.rho*g/g.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(lo,hi); controls=(cf.detach().cpu().numpy().reshape(-1)*ts+tm).reshape(fs,5)
                obs1=obs0
                for control in controls: obs1,_,term,_,_=env.step(control);
                frame=u.render().copy(); i=out['action'].shape[0]
                for d in out.values(): d.resize(i+1,axis=0)
                out['pixels'][i]=np.stack([start,frame]); out['action'][i]=controls.reshape(-1); out['observation'][i]=np.stack([obs0,np.asarray(obs1,dtype=np.float32)]); out['source_index'][i]=row; out['hinge'][i]=hg
                if count%100==0 or count==len(rows): out.flush(); print(f'executed {count:,}/{len(rows):,}',flush=True)
        finally: out.close(); env.close()
    print(f'Wrote {a.output}')
if __name__=='__main__': main()

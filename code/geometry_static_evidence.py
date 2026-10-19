"""Geometry-conditioned static soft-order evidence for short signal blocks."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter,hilbert,sosfiltfilt
from scipy.stats import kurtosis

def _ratio(envelope,sample_rate,rpm,order):
 x=envelope-envelope.mean();power=np.abs(np.fft.rfft(x))**2;frequency=np.fft.rfftfreq(len(x),1/sample_rate);resolution=sample_rate/len(x);values=[]
 for harmonic in range(1,5):
  target=harmonic*order*rpm/60.;sigma=max(2*resolution,.08*target);core=np.exp(-.5*((frequency-target)/sigma)**2);wide=np.exp(-.5*((frequency-target)/(4*sigma))**2);background=np.clip(wide-core,0,None);cp=float(np.sum(core*power)/max(np.sum(core),1e-12));bp=float(np.sum(background*power)/max(np.sum(background),1e-12));values.append(np.log1p(cp/max(bp,1e-12)))
 return float(np.mean(values))

def block_evidence(raw_blocks,sample_rate,rpms,orders,bands):
 raw_blocks=np.asarray(raw_blocks,dtype=np.float64);rpms=np.broadcast_to(np.asarray(rpms,dtype=float),len(raw_blocks));nyquist=sample_rate/2.;all_bands=[]
 for low,high in bands:
  sos=butter(4,(low/nyquist,high/nyquist),btype='bandpass',output='sos');filtered=sosfiltfilt(sos,raw_blocks,axis=-1);envelopes=np.abs(hilbert(filtered,axis=-1));rows=[]
  for block,envelope,rpm in zip(filtered,envelopes,rpms):rows.append([*[_ratio(envelope,sample_rate,rpm,o) for o in orders],float(kurtosis(block,fisher=True,bias=False)),float(np.log1p(np.sqrt(np.mean(block**2))))])
  all_bands.append(rows)
 out=np.asarray(all_bands,dtype=np.float32).transpose(1,0,2)
 if not np.isfinite(out).all():raise RuntimeError('non-finite geometry evidence')
 return out # [block,band,order+2]

def bag_descriptor(evidence,order_count):
 evidence=np.asarray(evidence);features=[]
 for band in range(evidence.shape[1]):
  for column in range(evidence.shape[2]):
   value=evidence[:,band,column];features.extend([float(value.mean()),float(np.median(value))])
 for order in range(order_count):
  best=evidence[:,:,order].max(1);features.extend([float(best.mean()),float(np.median(best))])
 out=np.asarray(features,dtype=np.float32)
 if not np.isfinite(out).all():raise RuntimeError('non-finite bag descriptor')
 return out

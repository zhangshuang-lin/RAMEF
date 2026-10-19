"""Physics-informed U2-STFT models for PU few-shot diagnosis.

The implementation intentionally keeps the original RSU/U2 backbone while adding
an order-synchronous complex STFT, sparse cyclic coherence, phase increments,
low-rank transfer counterfactuals, and uncertainty-aware prototypes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, 2, 2, ceil_mode=True)
        return super().forward(x)


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        return super().forward(torch.cat([x1, x2], dim=1))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        if height < 2:
            raise ValueError("RSU height must be at least 2")
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        enc = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        dec = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            enc.append(DownConvBNReLU(mid_ch, mid_ch))
            dec.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        enc.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules, self.decode_modules = nn.ModuleList(enc), nn.ModuleList(dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in, saved = self.conv_in(x), []
        x = x_in
        for module in self.encode_modules:
            x = module(x); saved.append(x)
        x = saved.pop()
        for module in self.decode_modules:
            x = module(x, saved.pop())
        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([
            ConvBNReLU(out_ch, mid_ch), ConvBNReLU(mid_ch, mid_ch, dilation=2),
            ConvBNReLU(mid_ch, mid_ch, dilation=4), ConvBNReLU(mid_ch, mid_ch, dilation=8)])
        self.decode_modules = nn.ModuleList([
            ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
            ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
            ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in, saved = self.conv_in(x), []
        x = x_in
        for module in self.encode_modules:
            x = module(x); saved.append(x)
        x = saved.pop()
        for module in self.decode_modules:
            x = module(torch.cat([x, saved.pop()], dim=1))
        return x + x_in


class U2Encoder(nn.Module):
    def __init__(self, cfg: List[List[Union[int, bool]]]):
        super().__init__()
        self.out_channels, modules = [], []
        for height, in_ch, mid_ch, out_ch, use_rsu4f in cfg:
            modules.append(RSU4F(in_ch, mid_ch, out_ch) if use_rsu4f
                           else RSU(height, in_ch, mid_ch, out_ch))
            self.out_channels.append(out_ch)
        self.encode_modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, return_stages: bool = False):
        stages = []
        for i, block in enumerate(self.encode_modules):
            x = block(x); stages.append(x)
            if i != len(self.encode_modules) - 1:
                x = F.max_pool2d(x, 2, 2, ceil_mode=True)
        return (x, stages) if return_stages else x


class ComplexSTFT(nn.Module):
    """Complex STFT. For L=8192, center=True gives 33 (not 29) frames."""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256,
                 win_length: int = 1024, center: bool = True):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length, self.center = (
            n_fft, hop_length, win_length, center)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [B,L] or [B,1,L]")
        return torch.stft(waveform, self.n_fft, self.hop_length, self.win_length,
                          self.window.to(waveform), self.center, return_complex=True)


class OrderResample(nn.Module):
    """Map Hz to a fixed order grid, removing the deterministic speed scaling."""
    def __init__(self, sample_rate: float = 64000.0, n_fft: int = 1024,
                 order_min: float = 0.5, order_max: float = 256.0,
                 order_bins: int = 256):
        super().__init__()
        self.sample_rate, self.n_fft = sample_rate, n_fft
        self.register_buffer("order_axis", torch.linspace(order_min, order_max, order_bins))

    def forward(self, spectrum: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(spectrum):
            raise TypeError("OrderResample expects a complex STFT")
        b, f, t = spectrum.shape
        rpm = rpm.reshape(b).to(spectrum.real).clamp_min(1.0)
        target_hz = self.order_axis[None].to(spectrum.real) * (rpm[:, None] / 60.0)
        max_hz = self.sample_rate / 2.0
        # grid_sample x-coordinate indexes frequency; y-coordinate indexes time.
        grid_x = target_hz / max_hz * 2.0 - 1.0
        grid_y = torch.linspace(-1, 1, t, device=spectrum.device, dtype=spectrum.real.dtype)
        gx = grid_x[:, None, :].expand(b, t, -1)
        gy = grid_y[None, :, None].expand(b, -1, self.order_axis.numel())
        grid = torch.stack([gx, gy], dim=-1)
        inp = torch.stack([spectrum.real, spectrum.imag], dim=1)
        out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
        return torch.complex(out[:, 0].transpose(1, 2), out[:, 1].transpose(1, 2))


def _fractional_order_shift(x: torch.Tensor, shift_bins: torch.Tensor) -> torch.Tensor:
    """Differentiably sample x(o+shift,t) along the order axis."""
    b, o, t = x.shape
    base = torch.arange(o, device=x.device, dtype=x.real.dtype)[None, :] + shift_bins[:, None]
    gx = torch.linspace(-1, 1, t, device=x.device, dtype=x.real.dtype)[None, :, None].expand(b, -1, o)
    gy = (base / max(o - 1, 1) * 2 - 1)[:, None, :].expand(b, t, -1)
    grid = torch.stack([gx, gy], dim=-1)
    inp = torch.stack([x.real, x.imag], dim=1)
    y = F.grid_sample(inp, grid, align_corners=True, padding_mode="zeros")
    return torch.complex(y[:, 0].transpose(1, 2), y[:, 1].transpose(1, 2))


class SparseCyclicCoherence(nn.Module):
    """Sparse spectral-correlation slices at physically specified cyclic orders."""
    def __init__(self, cyclic_orders: Sequence[float], order_step: float, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("cyclic_orders", torch.tensor(cyclic_orders, dtype=torch.float32))
        self.order_step, self.eps = float(order_step), eps

    def forward(self, x_order: torch.Tensor) -> torch.Tensor:
        b = x_order.shape[0]
        results = []
        for alpha in self.cyclic_orders.to(x_order.real):
            half_shift = torch.full((b,), float(alpha / (2 * self.order_step)),
                                    device=x_order.device, dtype=x_order.real.dtype)
            plus = _fractional_order_shift(x_order, half_shift)
            minus = _fractional_order_shift(x_order, -half_shift)
            cross = (plus * minus.conj()).mean(dim=-1).abs()
            power = (plus.abs().square().mean(dim=-1) *
                     minus.abs().square().mean(dim=-1)).sqrt()
            results.append(cross / power.clamp_min(self.eps))
        return torch.stack(results, dim=1).clamp(0, 1)  # [B,A,O]


class RefinedCyclicCoherence(nn.Module):
    """Soft PU-6203 cyclic anchors with local Gaussian order support.

    Close BPFO+1 and BPFI-1 anchors are merged at 4.0 orders. Gaussian support
    avoids treating small slip/estimation errors as a different physical state.
    """
    def __init__(self, anchors: Sequence[float], order_axis: torch.Tensor,
                 order_step: float, sigma: float = .2, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("cyclic_orders", torch.tensor(anchors, dtype=torch.float32))
        self.register_buffer("order_axis", order_axis.detach().float().clone())
        self.order_step, self.sigma, self.eps = float(order_step), float(sigma), eps

    def forward(self, x_order: torch.Tensor) -> torch.Tensor:
        b = x_order.shape[0]; results = []
        axis = self.order_axis.to(x_order.real)
        # Average three nearby cyclic separations for a differentiable soft
        # tolerance; the spatial Gaussian retains evidence local to the anchor.
        offsets = (-self.sigma, 0.0, self.sigma)
        offset_weights = x_order.real.new_tensor([.25, .5, .25])
        for anchor in self.cyclic_orders.to(x_order.real):
            local = 0.0
            for weight, offset in zip(offset_weights, offsets):
                alpha = anchor + offset
                shift = torch.full((b,), float(alpha / (2 * self.order_step)),
                                   device=x_order.device, dtype=x_order.real.dtype)
                plus = _fractional_order_shift(x_order, shift)
                minus = _fractional_order_shift(x_order, -shift)
                cross = (plus * minus.conj()).mean(dim=-1).abs()
                power = (plus.abs().square().mean(dim=-1) *
                         minus.abs().square().mean(dim=-1)).sqrt()
                local = local + weight * cross / power.clamp_min(self.eps)
            window = torch.exp(-.5 * ((axis - anchor) / self.sigma) ** 2)
            results.append(local * window[None])
        return torch.stack(results, dim=1).clamp(0, 1)


class PhaseIncrementFrontEnd(nn.Module):
    """Wrap-free time phase increment and group-delay-like frequency increment."""
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, groups=4, bias=False), nn.BatchNorm2d(8),
            nn.ReLU(inplace=True), nn.Conv2d(8, out_channels, 1))

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        dt = torch.angle(x[:, :, 1:] * x[:, :, :-1].conj())
        df = torch.angle(x[:, 1:, :] * x[:, :-1, :].conj())
        dt = F.pad(dt, (0, 1)); df = F.pad(df, (0, 0, 0, 1))
        phase = torch.stack([dt.cos(), dt.sin(), df.cos(), df.sin()], dim=1)
        phase = F.interpolate(phase, size=size, mode="bilinear", align_corners=False)
        return self.net(phase)


class PhysicsFrontEnd(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024,
                 sample_rate=64000, order_min=.5, order_max=256., order_bins=256,
                 cyclic_orders=(.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.), out_size=(128, 64),
                 use_refined_cyclic_anchors: bool = False,
                 refined_cyclic_anchors=(1., 2.05, 3.05, 4., 4.95, 5.95),
                 cyclic_anchor_sigma: float = .2, late_fusion: bool = False,
                 frontend_ablation: str = "full"):
        super().__init__()
        self.stft = ComplexSTFT(n_fft, hop_length, win_length, True)
        self.order = OrderResample(sample_rate, n_fft, order_min, order_max, order_bins)
        step = (order_max - order_min) / (order_bins - 1)
        if use_refined_cyclic_anchors:
            self.scoc = RefinedCyclicCoherence(
                refined_cyclic_anchors, self.order.order_axis, step, cyclic_anchor_sigma)
        else:
            self.scoc = SparseCyclicCoherence(cyclic_orders, step)
        self.phase = PhaseIncrementFrontEnd(1)
        self.cyclic_compress = nn.Conv2d(len(self.scoc.cyclic_orders), 3, 1)
        allowed = {"hz", "hz_order", "hz_order_cyclic", "full"}
        if frontend_ablation not in allowed:
            raise ValueError(f"frontend_ablation must be one of {sorted(allowed)}")
        self.frontend_ablation = frontend_ablation
        self.late_fusion = late_fusion
        # order magnitude (speed-normalized) + full-Hz magnitude (resonance
        # carrier) + 3 cyclic channels + 1 phase-increment channel.
        self.fuse = nn.Conv2d(6, 3, 1)
        if late_fusion:
            # Encode physically different axes separately before interaction.
            self.order_encoder = nn.Sequential(ConvBNReLU(5, 8), nn.Conv2d(8, 3, 1))
            self.hz_encoder = nn.Sequential(ConvBNReLU(1, 8), nn.Conv2d(8, 3, 1))
            self.cross_h = nn.Conv2d(3, 3, 1); self.cross_o = nn.Conv2d(3, 3, 1)
            self.late_map_fuse = nn.Conv2d(9, 3, 1)
        self.out_size = tuple(out_size)

    def maps_from_order(self, x_order: torch.Tensor, x_hz: Optional[torch.Tensor] = None):
        magnitude = torch.log1p(x_order.abs())[:, None]
        if self.frontend_ablation in {"hz_order_cyclic", "full"}:
            coherence = self.scoc(x_order)
            cyc_map = coherence[:, :, :, None].expand(-1, -1, -1, x_order.shape[-1])
            cyc_map = self.cyclic_compress(cyc_map)
        else:
            coherence = x_order.real.new_ones(
                x_order.shape[0], len(self.scoc.cyclic_orders), x_order.shape[1])
            cyc_map = x_order.real.new_zeros(x_order.shape[0], 3, *x_order.shape[-2:])
        phase_source = x_hz if x_hz is not None else x_order
        phase = (self.phase(phase_source, x_order.shape[-2:])
                 if self.frontend_ablation == "full"
                 else x_order.real.new_zeros(x_order.shape[0], 1, *x_order.shape[-2:]))
        # Retain the entire 0-32 kHz carrier spectrum. Order normalization alone
        # at 256 orders reaches only 3.84 kHz for 900 rpm and would discard the
        # high-frequency bearing resonance that carries impulsive evidence.
        hz_magnitude = torch.log1p(phase_source.abs())[:, None]
        hz_magnitude = F.interpolate(hz_magnitude, x_order.shape[-2:], mode="bilinear", align_corners=False)
        if self.frontend_ablation == "hz":
            magnitude = torch.zeros_like(magnitude)
        if self.late_fusion:
            order_map = self.order_encoder(torch.cat([magnitude, cyc_map, phase], dim=1))
            hz_map = self.hz_encoder(hz_magnitude)
            cross = self.cross_h(hz_map) * self.cross_o(order_map)
            fused = self.late_map_fuse(torch.cat([hz_map, order_map, cross], dim=1))
        else:
            fused = self.fuse(torch.cat([magnitude, hz_magnitude, cyc_map, phase], dim=1))
        fused = F.interpolate(fused, self.out_size, mode="bilinear", align_corners=False)
        mean = fused.mean((2, 3), keepdim=True)
        std = fused.std((2, 3), keepdim=True).clamp_min(1e-5)
        return (fused - mean) / std, coherence

    def forward(self, waveform: torch.Tensor, rpm: torch.Tensor):
        x_hz = self.stft(waveform)
        x_order = self.order(x_hz, rpm)
        fused, coherence = self.maps_from_order(x_order, x_hz)
        return fused, x_order, coherence


class LongContextLowOrderBranch(nn.Module):
    """Low-cost 0-12 order envelope spectra from four long-context blocks."""
    def __init__(self, sample_rate=64000, long_window_length=32768,
                 order_bins=128, order_max=12.,
                 bands=((500.,2500.),(2500.,6000.),(6000.,12000.)),
                 output_dim=128):
        super().__init__()
        if long_window_length % 4:
            raise ValueError("long_window_length must be divisible by four")
        self.sample_rate=float(sample_rate); self.long_window_length=long_window_length
        self.block_length=long_window_length//4; self.bands=tuple(tuple(x) for x in bands)
        self.register_buffer("low_order_axis",torch.linspace(.25,order_max,order_bins))
        self.temporal_encoder=nn.Sequential(
            nn.Conv1d(4,16,5,padding=2,bias=False),nn.BatchNorm1d(16),nn.ReLU(inplace=True),
            nn.Conv1d(16,32,3,padding=1,bias=False),nn.BatchNorm1d(32),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))
        self.output=nn.Linear(32,output_dim)

    @staticmethod
    def _analytic_envelope(filtered):
        n=filtered.shape[-1]; spectrum=torch.fft.fft(filtered.float(),dim=-1)
        h=torch.zeros(n,device=filtered.device,dtype=filtered.dtype)
        h[0]=1
        if n%2==0: h[n//2]=1; h[1:n//2]=2
        else: h[1:(n+1)//2]=2
        return torch.fft.ifft(spectrum*h,dim=-1).abs()

    def order_maps(self,long_waveform,rpm):
        b=long_waveform.shape[0]
        blocks=long_waveform.float().reshape(b,4,self.block_length)
        freq=torch.fft.fftfreq(self.block_length,1/self.sample_rate,device=blocks.device)
        maps=[]
        for lo,hi in self.bands:
            mask=((freq.abs()>=lo)&(freq.abs()<=hi)).to(blocks.dtype)
            filtered=torch.fft.ifft(torch.fft.fft(blocks,dim=-1)*mask,dim=-1).real
            env=self._analytic_envelope(filtered)
            env=env-env.mean(-1,keepdim=True)
            spec=torch.fft.rfft(env*torch.hann_window(self.block_length,device=env.device),dim=-1).abs()
            target_hz=self.low_order_axis[None].to(spec)*(rpm.reshape(b,1).to(spec)/60.)
            bin_pos=target_hz/(self.sample_rate/self.block_length)
            lo_idx=bin_pos.floor().long().clamp(0,spec.shape[-1]-1)
            hi_idx=(lo_idx+1).clamp(0,spec.shape[-1]-1)
            frac=(bin_pos-lo_idx).to(spec.dtype)
            lo_idx=lo_idx[:,None].expand(-1,4,-1); hi_idx=hi_idx[:,None].expand(-1,4,-1)
            sampled=spec.gather(2,lo_idx)*(1-frac[:,None])+spec.gather(2,hi_idx)*frac[:,None]
            maps.append(torch.log1p(sampled))
        result=torch.stack(maps).mean(0)
        mean=result.mean(-1,keepdim=True); std=result.std(-1,keepdim=True).clamp_min(1e-5)
        return (result-mean)/std

    def forward(self,long_waveform,rpm):
        maps=self.order_maps(long_waveform,rpm)
        return self.output(self.temporal_encoder(maps).squeeze(-1)),maps


class LongContextIRORPhysicsBranch(nn.Module):
    """Long-context BPFI/BPFO family evidence with four-block persistence.

    Structural resonance bands are enveloped independently. Complex envelope
    spectra are sampled at soft theoretical order anchors. The representation
    retains mean evidence, block variability and phase stability, specifically
    targeting the rotating-load-zone modulation that separates IR from OR.
    """
    def __init__(self,sample_rate=64000,long_window_length=32768,
                 anchors=(1.,2.05,3.05,4.05,3.95,4.95,5.95),
                 anchor_sigma=.25,bands=((500.,2500.),(2500.,6000.),(6000.,12000.)),
                 num_orders=8,feature_dim=128,use_temporal_stability=True):
        super().__init__()
        if long_window_length%4: raise ValueError("long window must divide into four blocks")
        self.sample_rate=float(sample_rate); self.long_window_length=int(long_window_length)
        self.block_length=self.long_window_length//4; self.bands=tuple(tuple(b) for b in bands)
        self.register_buffer("anchors",torch.tensor(anchors,dtype=torch.float32))
        self.anchor_sigma=float(anchor_sigma); self.use_temporal_stability=use_temporal_stability
        self.num_orders=num_orders; self.feature_dim=feature_dim
        self.encoder=nn.Sequential(nn.Linear(23,32),nn.GELU(),nn.Linear(32,num_orders*feature_dim))
        # Exact identity at initialization when added to the frozen D tokens.
        nn.init.zeros_(self.encoder[-1].weight); nn.init.zeros_(self.encoder[-1].bias)

    @staticmethod
    def _analytic_envelope(filtered):
        n=filtered.shape[-1]; spectrum=torch.fft.fft(filtered.float(),dim=-1)
        h=torch.zeros(n,device=filtered.device,dtype=filtered.dtype); h[0]=1
        if n%2==0: h[n//2]=1; h[1:n//2]=2
        else: h[1:(n+1)//2]=2
        return torch.fft.ifft(spectrum*h,dim=-1).abs()

    def evidence(self,long_waveform,rpm):
        b=long_waveform.shape[0]; blocks=long_waveform.float().reshape(b,4,self.block_length)
        frequency=torch.fft.fftfreq(self.block_length,1/self.sample_rate,device=blocks.device)
        window=torch.hann_window(self.block_length,device=blocks.device)
        band_samples=[]; band_magnitudes=[]
        for lo,hi in self.bands:
            mask=((frequency.abs()>=lo)&(frequency.abs()<=hi)).to(blocks.dtype)
            filtered=torch.fft.ifft(torch.fft.fft(blocks,dim=-1)*mask,dim=-1).real
            envelope=self._analytic_envelope(filtered); envelope=envelope-envelope.mean(-1,keepdim=True)
            spectrum=torch.fft.rfft(envelope*window,dim=-1)
            # Soft anchor integration in Hz corresponds to a fixed order width.
            hz=torch.fft.rfftfreq(self.block_length,1/self.sample_rate,device=blocks.device)
            center=self.anchors[None,:,None].to(hz)*(rpm[:,None,None].to(hz)/60.)
            # A requested soft band narrower than one FFT bin is not physically
            # resolvable; enforce the long-block spectral resolution.
            sigma=(self.anchor_sigma*rpm[:,None,None].to(hz)/60.).clamp_min(
                self.sample_rate/self.block_length)
            weights=torch.exp(-.5*((hz[None,None,:]-center)/sigma)**2)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
            sampled=torch.einsum("baf,btf->bta",weights.to(spectrum.dtype),spectrum)
            band_samples.append(sampled)
            band_magnitudes.append(torch.einsum("baf,btf->bta",weights,spectrum.abs()))
        complex_evidence=torch.stack(band_samples,dim=1) # B,bands,blocks,anchors
        magnitude=torch.log1p(torch.stack(band_magnitudes,dim=1)).mean(1) # B,blocks,anchors
        mean=magnitude.mean(1); variation=magnitude.std(1,unbiased=False)
        unit=complex_evidence/complex_evidence.abs().clamp_min(1e-6)
        phase_stability=unit.mean((1,2)).abs()
        if not self.use_temporal_stability:
            variation=torch.zeros_like(variation); phase_stability=torch.zeros_like(phase_stability)
        # anchors indices: 1x, BPFO-1, BPFO, BPFO+1, BPFI-1, BPFI, BPFI+1
        ir_ratio=torch.logsumexp(mean[:,[4,6]],1)-mean[:,5]-torch.log(mean.new_tensor(2.))
        or_ratio=mean[:,2]-torch.logsumexp(mean[:,[1,3]],1)+torch.log(mean.new_tensor(2.))
        features=torch.cat((mean,variation,phase_stability,ir_ratio[:,None],or_ratio[:,None]),1)
        return features,{"anchor_magnitude":magnitude,"phase_stability":phase_stability,
                         "ir_sideband_ratio":ir_ratio,"or_central_ratio":or_ratio}

    def forward(self,long_waveform,rpm):
        features,details=self.evidence(long_waveform,rpm)
        tokens=self.encoder(features).view(-1,self.num_orders,self.feature_dim)
        return tokens,features,details


class GateNet(nn.Module):
    """Seven-statistic support-only scalar fusion gate."""
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(7,8),nn.ReLU(),nn.Linear(8,1))
    def forward(self,stats): return torch.sigmoid(self.net(stats))


class SpeedResidualAdapter(nn.Module):
    """Zero-initialized low-rank token correction conditioned on known rpm.

    At initialization the module is exactly the identity, so loading an E0
    checkpoint cannot alter its geometry before source-only adaptation.
    """
    def __init__(self, feature_dim: int, bottleneck_ratio: int = 8,
                 reference_rpm: float = 1500.):
        super().__init__()
        hidden=max(4,feature_dim//bottleneck_ratio)
        self.down=nn.Linear(feature_dim,hidden)
        self.up=nn.Linear(hidden,feature_dim)
        self.reference_rpm=float(reference_rpm)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self,tokens,rpm):
        speed=torch.log(rpm.reshape(-1,1,1).to(tokens).clamp_min(1.)/
                        self.reference_rpm)
        residual=self.up(F.gelu(self.down(tokens)))
        return tokens+speed*residual


class LowRankTransferCounterfactual(nn.Module):
    """A low-rank smooth complex transfer correction in order-time coordinates."""
    def __init__(self, feature_channels: int, order_bins: int = 256,
                 time_basis: int = 16, rank: int = 4, latent_dim: int = 64):
        super().__init__()
        self.rank, self.order_bins, self.time_basis = rank, order_bins, time_basis
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fault_head = nn.Linear(feature_channels, latent_dim)
        self.transfer_head = nn.Linear(feature_channels, latent_dim)
        self.coeff = nn.Linear(latent_dim, rank * 2)
        self.freq_gain = nn.Parameter(torch.randn(rank, order_bins) * .01)
        self.freq_phase = nn.Parameter(torch.randn(rank, order_bins) * .01)
        self.time_gain = nn.Parameter(torch.randn(rank, time_basis) * .01)
        self.time_phase = nn.Parameter(torch.randn(rank, time_basis) * .01)

    def encode(self, early_feature: torch.Tensor):
        h = self.pool(early_feature).flatten(1)
        return self.fault_head(h), self.transfer_head(h)

    def correction(self, z_transfer: torch.Tensor, time_frames: int):
        c = self.coeff(z_transfer).view(-1, 2, self.rank)
        tg = F.interpolate(self.time_gain[None], time_frames, mode="linear", align_corners=False)[0]
        tp = F.interpolate(self.time_phase[None], time_frames, mode="linear", align_corners=False)[0]
        gain_basis = self.freq_gain[:, :, None] * tg[:, None, :]
        phase_basis = self.freq_phase[:, :, None] * tp[:, None, :]
        gain = torch.einsum("br,rot->bot", c[:, 0], gain_basis).clamp(-2, 2)
        phase = torch.einsum("br,rot->bot", c[:, 1], phase_basis).clamp(-torch.pi, torch.pi)
        return gain, phase

    def apply(self, x_order: torch.Tensor, z_transfer: torch.Tensor):
        gain, phase = self.correction(z_transfer, x_order.shape[-1])
        # Keep complex transfer arithmetic in complex64 even under AMP; PyTorch
        # complex-half support is incomplete and is unnecessary at this size.
        gain, phase = gain.float(), phase.float()
        corrected = x_order.to(torch.complex64) * torch.exp(gain) * torch.exp(1j * phase)
        return corrected, gain, phase


class CyclicOrderTokenizer(nn.Module):
    def forward(self, feature: torch.Tensor, coherence: torch.Tensor) -> torch.Tensor:
        # Resize the U2 map to the physical order grid. Each cyclic slice weights
        # order positions; time is averaged only after local evidence is retained.
        f = F.interpolate(feature, size=(coherence.shape[-1], feature.shape[-1]),
                          mode="bilinear", align_corners=False).mean(dim=-1)
        weights = coherence / coherence.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bao,bco->bac", weights, f)


@dataclass
class PrototypeState:
    mean: torch.Tensor
    variance: torch.Tensor
    classes: torch.Tensor


class CyclicPrototypeClassifier(nn.Module):
    def __init__(self, num_classes: int, num_orders: int, feature_dim: int,
                 variance_floor: float = .05):
        super().__init__()
        self.num_classes, self.variance_floor = num_classes, variance_floor
        self.global_prototypes = nn.Parameter(torch.randn(num_classes, num_orders, feature_dim) * .02)
        self.log_precision = nn.Parameter(torch.zeros(num_classes, num_orders, feature_dim))

    def fit_support(self, tokens: torch.Tensor, labels: torch.Tensor,
                    bayesian: bool = False) -> PrototypeState:
        means, variances, classes = [], [], torch.arange(self.num_classes, device=tokens.device)
        for c in classes:
            z = tokens[labels == c]
            if z.numel() == 0:
                means.append(self.global_prototypes[c]); variances.append(torch.ones_like(self.global_prototypes[c]))
                continue
            sample_mean = z.mean(0)
            sample_var = z.var(0, unbiased=False) + self.variance_floor
            if bayesian:
                prior_precision = self.log_precision[c].exp().clamp(1e-3, 1e3)
                data_precision = z.shape[0] / sample_var
                post = (prior_precision * self.global_prototypes[c] + data_precision * sample_mean) / (
                    prior_precision + data_precision)
                means.append(post); variances.append(1.0 / (prior_precision + data_precision))
            else:
                means.append(sample_mean); variances.append(sample_var)
        return PrototypeState(torch.stack(means), torch.stack(variances), classes)

    def logits(self, query: torch.Tensor, state: PrototypeState) -> torch.Tensor:
        diff = query[:, None] - state.mean[None]
        distance = (diff.square() / state.variance[None].clamp_min(self.variance_floor)).mean((2, 3))
        return -distance


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength): ctx.strength = strength; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.strength * grad, None


class U2STFT(nn.Module):
    """Complete OSC-Proto U2-STFT model with optional counterfactual branch."""
    def __init__(self, num_classes: int = 4, lite: bool = True,
                 use_low_order_branch: bool = False,
                 support_conditioned_gate: bool = False,
                 long_window_length: int = 32768,
                 low_order_bins: int = 128,
                 low_order_max: float = 12.,
                 low_order_bands=((500.,2500.),(2500.,6000.),(6000.,12000.)),
                 use_speed_adapter: bool = False,
                 speed_adapter_ratio: int = 8,
                 speed_reference_rpm: float = 1500.,
                 use_iror_physics_branch: bool = False,
                 physics_use_temporal_stability: bool = True,
                 physics_anchors=(1.,2.05,3.05,4.05,3.95,4.95,5.95),
                 physics_anchor_sigma: float = .25,
                 physics_fusion_max: float = .25,
                 **frontend_kwargs):
        super().__init__()
        self.frontend_name = "order_stft_scoc_phase"
        self.frontend = PhysicsFrontEnd(**frontend_kwargs)
        width = 16 if lite else 32
        cfg = [[7, 3, width, width * 2, False], [6, width * 2, width, width * 4, False],
               [5, width * 4, width * 2, width * 8, False],
               [4, width * 8, width * 4, width * 8, True]]
        self.encoder = U2Encoder(cfg)
        self.out_channels = self.encoder.out_channels[-1]
        self.tokenizer = CyclicOrderTokenizer()
        a = len(self.frontend.scoc.cyclic_orders)
        self.prototype = CyclicPrototypeClassifier(num_classes, a, self.out_channels)
        self.linear_classifier = nn.Linear(a * self.out_channels, num_classes)
        self.counterfactual = LowRankTransferCounterfactual(self.encoder.out_channels[0])
        self.domain_classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))
        self.use_speed_adapter=use_speed_adapter
        self.speed_adapter=(SpeedResidualAdapter(self.out_channels,speed_adapter_ratio,
                                                 speed_reference_rpm)
                            if use_speed_adapter else None)
        self.use_iror_physics_branch=use_iror_physics_branch
        self.physics_fusion_max=float(physics_fusion_max)
        if use_iror_physics_branch:
            self.iror_physics=LongContextIRORPhysicsBranch(
                sample_rate=frontend_kwargs.get("sample_rate",64000),
                long_window_length=long_window_length,anchors=physics_anchors,
                anchor_sigma=physics_anchor_sigma,bands=low_order_bands,
                num_orders=a,feature_dim=self.out_channels,
                use_temporal_stability=physics_use_temporal_stability)
            self.physics_gate_logit=nn.Parameter(torch.tensor(-2.944439)) # sigmoid=.05
        self.use_low_order_branch=use_low_order_branch
        self.support_conditioned_gate=support_conditioned_gate
        if use_low_order_branch:
            self.low_order=LongContextLowOrderBranch(
                sample_rate=frontend_kwargs.get("sample_rate",64000),
                long_window_length=long_window_length,order_bins=low_order_bins,
                order_max=low_order_max,bands=low_order_bands,output_dim=self.out_channels)
            self.low_to_tokens=nn.Linear(self.out_channels,a*self.out_channels)
            self.global_gate_logit=nn.Parameter(torch.tensor(-1.3862944)) # sigmoid=0.2
            self.gate_net=GateNet() if support_conditioned_gate else None

    def _low_tokens(self,long_waveform,rpm):
        low,maps=self.low_order(long_waveform,rpm)
        tokens=self.low_to_tokens(low).view(-1,len(self.frontend.scoc.cyclic_orders),self.out_channels)
        return tokens,maps

    def fuse_token_branches(self,main,low,gate=None):
        if low is None: return main
        if gate is None: gate=torch.sigmoid(self.global_gate_logit)
        while torch.is_tensor(gate) and gate.ndim<main.ndim: gate=gate.unsqueeze(-1)
        return (1-gate)*main+gate*low

    def support_gate(self,main_support,low_support,labels):
        delta=(low_support-main_support).flatten(1); stats=[]; class_means=[]
        for c in range(self.prototype.num_classes):
            z=delta[labels==c]; mean=z.mean(0); class_means.append(mean)
            stats.extend([mean.mean(),z.var(0,unbiased=False).mean()])
        stats.append((class_means[1]-class_means[2]).square().mean().sqrt())
        return self.gate_net(torch.stack(stats)[None]).reshape(1)

    def encode_order(self, x_order: torch.Tensor, x_hz: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        image, coherence = self.frontend.maps_from_order(x_order, x_hz)
        feature, stages = self.encoder(image, return_stages=True)
        tokens = self.tokenizer(feature, coherence)
        z_fault, z_transfer = self.counterfactual.encode(stages[0])
        return {"feature": feature, "stages": stages, "tokens": tokens,
                "embedding": tokens.flatten(1), "coherence": coherence,
                "z_fault": z_fault, "z_transfer": z_transfer, "image": image}

    def forward(self, waveform: torch.Tensor, rpm: torch.Tensor,
                domain_grl: float = 0.0,
                long_waveform: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x_hz = self.frontend.stft(waveform)
        x_order = self.frontend.order(x_hz, rpm)
        out = self.encode_order(x_order, x_hz)
        out["main_tokens"]=out["tokens"]
        if self.use_low_order_branch:
            if long_waveform is None:
                long_waveform=waveform.repeat(1,self.low_order.long_window_length//waveform.shape[-1])
            low_tokens,low_maps=self._low_tokens(long_waveform,rpm)
            out["low_tokens"],out["low_order_maps"]=low_tokens,low_maps
            out["tokens"]=self.fuse_token_branches(out["main_tokens"],low_tokens)
            out["embedding"]=out["tokens"].flatten(1)
        out["pre_adapter_tokens"]=out["tokens"]
        if self.speed_adapter is not None:
            out["tokens"]=self.speed_adapter(out["tokens"],rpm)
            out["embedding"]=out["tokens"].flatten(1)
        if self.use_iror_physics_branch:
            if long_waveform is None:
                repeats=math.ceil(self.iror_physics.long_window_length/waveform.shape[-1])
                long_waveform=waveform.repeat(1,repeats)[:,:self.iror_physics.long_window_length]
            physics_tokens,physics_features,physics_details=self.iror_physics(long_waveform,rpm)
            beta=self.physics_fusion_max*torch.sigmoid(self.physics_gate_logit)
            out["pre_physics_tokens"]=out["tokens"]
            out["physics_tokens"],out["physics_features"]=physics_tokens,physics_features
            out["physics_details"],out["physics_beta"]=physics_details,beta
            out["tokens"]=out["tokens"]+beta*physics_tokens
            out["embedding"]=out["tokens"].flatten(1)
        out["x_hz"], out["x_order"] = x_hz, x_order
        out["rpm"] = rpm
        out["logits"] = self.linear_classifier(out["embedding"])
        reversed_fault = GradientReversal.apply(out["z_fault"], domain_grl)
        out["domain_logits"] = self.domain_classifier(reversed_fault)
        return out

    def make_counterfactual(self, out: Dict[str, torch.Tensor], permutation: torch.Tensor):
        x_cf, gain, phase = self.counterfactual.apply(
            out["x_order"], out["z_transfer"][permutation])
        cf = self.encode_order(x_cf)
        cf["main_tokens"]=cf["tokens"]
        if self.use_low_order_branch and "low_tokens" in out:
            cf["low_tokens"]=out["low_tokens"]
            cf["tokens"]=self.fuse_token_branches(cf["main_tokens"],cf["low_tokens"])
            cf["embedding"]=cf["tokens"].flatten(1)
        cf["pre_adapter_tokens"]=cf["tokens"]
        if self.speed_adapter is not None:
            cf["tokens"]=self.speed_adapter(cf["tokens"],out["rpm"])
            cf["embedding"]=cf["tokens"].flatten(1)
        if self.use_iror_physics_branch and "physics_tokens" in out:
            beta=self.physics_fusion_max*torch.sigmoid(self.physics_gate_logit)
            cf["physics_tokens"]=out["physics_tokens"]
            cf["tokens"]=cf["tokens"]+beta*out["physics_tokens"]
            cf["embedding"]=cf["tokens"].flatten(1)
        cf["x_order"], cf["transfer_gain"], cf["transfer_phase"] = x_cf, gain, phase
        cf["logits"] = self.linear_classifier(cf["embedding"])
        return cf

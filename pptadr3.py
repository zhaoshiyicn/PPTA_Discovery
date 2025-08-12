import sys
import os
import glob
import pickle
import tqdm
import json
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc
import jax
jax.config.update('jax_enable_x64', True)

import jax.random
import jax.numpy as jnp

import discovery as ds
import discovery.flow as dsf
import discovery.models.nanograv as ds_nanograv
import discovery.samplers.numpyro as ds_numpyro
from discovery import signals, likelihood, matrix
from discovery.solar import theta_impact, AU_light_sec, AU_pc, make_solardm, make_chromaticdelay

from enterprise.pulsar import Pulsar as ePulsar

ppta_prior = {
            "(.*_)?rednoise_log10_A.*": [-20, -11],
            "(.*_)?rednoise_gamma.*": [0, 7],
            "(.*_)?dm_gp_log10_A": [-18, -11],
            "(.*_)?dm_gp_gamma": [0, 7],
            "(.*_)?dm_gp_alpha": [1, 3],
            "(.*_)?sw_n_earth": [0, 20],
            "(.*_)?gp_sw_log10_A": [-10, -3],
            "(.*_)?gp_sw_gamma": [-4, 4],
            "(.*_)?hf_noise_log10_A.*": [-18, -11],
            "(.*_)?hf_noise_gamma.*": [0, 7],
            "J1713\+0747_dmexp_1_t0": [54650, 54850],
            "J1713\+0747_dmexp_2_t0": [57400, 57600],
            "J0437-4715_dmexp_1_t0": [57000, 57200],
            "J1643-1224_dmexp_1_t0": [57000, 57200],
            "J2145-0750_dmexp_1_t0": [56250, 56450],
            "(.*_)?log10_Amp": [-10, -2],
            "(.*_)?tau": [0, 2.5],
            "(.*_)?idx": [-2, 2],
            "(.*_)?dm1yr_log10_Amp": [-10, -2],
            "(.*_)?dm1yr_phase": [0, 2*np.pi],
            "(.*_)?dmgauss_log10_Amp": [-10, -2],
            "(.*_)?dmgauss_epoch": [53800, 54000],
            "(.*_)?dmgauss_log10_sigma": [0, 3],
            "(.*_)?chrom_gp_log10_A.*": [-18, -11],
            "(.*_)?chrom_gp_gamma.*": [0, 7],
            "(.*_)?band_noise_low_log10_A.*": [-18, -11],
            "(.*_)?band_noise_low_gamma.*": [0, 7],
            "(.*_)?band_noise_mid_log10_A.*": [-18, -11],
            "(.*_)?band_noise_mid_gamma.*": [0, 7],
            "(.*_)?band_noise_high_log10_A.*": [-18, -11],
            "(.*_)?band_noise_high_gamma.*": [0, 7],
            r".*_group_noises_[A-Za-z0-9_]+_log10_A$": [-18, -11],
            r".*_group_noises_[A-Za-z0-9_]+_gamma$": [0, 7],
            "crn_log10_A.*": [-18, -11],
            "crn_gamma.*": [0, 7],
        }

models_dict = {
               'hf': ['J0437-4715', 'J1017-7156', 'J1022+1001', 'J1600-3053', 'J1713+0747', 'J1744-1134', 'J1909-3744', 'J2241-5236'],
               'bn_low': ['J0437-4715', 'J0613-0200', 'J1017-7156', 'J1045-4509', 'J1600-3053', 
                          'J1643-1224', 'J1713+0747', 'J1909-3744', 'J1939+2134'],
               'bn_mid': ['J0437-4715'],
               'bn_high': ['J0437-4715'],
               'group_noise': {'J0437-4715': ['UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG', 'CASPSR_40CM'],
                               'J1713+0747': ['UWL_sbA', 'UWL_sbE', 'UWL_sbF', 'WBCORR_10CM'],
                               'J1909-3744': ['CPSR2_50CM'],
                               'J1017-7156': ['UWL_sbA', 'UWL_sbD'],
                               'J1022+1001': ['UWL_sbE', 'UWL_sbH']},
               'group_ecorr': {'J0437-4715': ['UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG', 'CASPSR_40CM', 'PDFB_20CM'],
                               'J1713+0747': ['UWL_sbA', 'UWL_sbE', 'UWL_sbF', 'WBCORR_10CM', 'CPSR2_20CM'],
                               'J1909-3744': ['CPSR2_50CM', 'CASPSR_40CM', 'PDFB1_1433', 'PDFB1_early_20CM'],
                               'J1017-7156': ['UWL_sbA', 'UWL_sbD'],
                               'J1022+1001': ['UWL_sbE', 'UWL_sbH']},
               'chrom': ['J0437-4715', 'J0613-0200', 'J1017-7156', 'J1045-4509', 'J1600-3053', 'J1643-1224', 'J1939+2134'],
               'dm1yr': ['J0613-0200'],
               'sw': {'constant_with_gp': ['J0437-4715', 'J0711-6830', 'J0900-3144', 'J1024-0719', 'J1643-1224', 
                                           'J1713+0747', 'J1730-2304', 'J1744-1134', 'J1909-3744', 'J2145-0750'],
                      'uniform': ['J0030+0451', 'J0125-2327', 'J0613-0200', 'J0614-3329', 'J1017-7156', 'J1022+1001', 'J1045-4509',
                                  'J1125-6014', 'J1446-4701', 'J1545-4550', 'J1600-3053', 'J1603-7202', 'J1741+1351', 'J1824-2452A',
                                  'J1832-0836', 'J1857+0943', 'J1902-5105', 'J1933-6211', 'J1939+2134', 'J2124-3358', 'J2129-5721',
                                  'J2241-5236']},
               'dmgauss': ['J1603-7202'],
               'dm_exp': {'J1713+0747': 2,
                          'J0437-4715': 1,
                          'J1643-1224': 1,
                          'J2145-0750': 1}
                }

psrnames = ['J0030+0451',
            'J0125-2327',
            'J0437-4715',
            'J0613-0200',
            'J0614-3329',
            'J0711-6830', 
            'J0900-3144',
            'J1017-7156',
            'J1022+1001',
            'J1024-0719', 
            'J1045-4509',
            'J1125-6014',
            'J1446-4701', 
            'J1545-4550',
            'J1600-3053',
            'J1603-7202',
            'J1643-1224',
            'J1713+0747',
            'J1730-2304',
            #'J1741+1351', #trash
            'J1744-1134',
            #'J1824-2452A', #also trash
            'J1857+0943',
            'J1832-0836',
            'J1902-5105',
            'J1909-3744',
            'J1933-6211',
            'J1939+2134',
            'J2124-3358',
            'J2129-5721',
            'J2145-0750',
            'J2241-5236']


with open('/home/zhaosy/work/data/pptadr3/32psrs_DE438_dis.pkl', 'rb') as f:
    dpsrs = pickle.load(f)
psrs = [p for p in dpsrs if p.name in psrnames]

def selection_freqband(psr):
    freqs = psr.freqs             
    backend_flags = psr.backend_flags  
    arr = np.array(['UWL' in val for val in backend_flags])

    labels = np.full_like(freqs, '', dtype=object)

    labels[(freqs < 960) & ~arr] = '40CM'
    labels[(freqs >= 960) & (freqs < 2048) & ~arr] = '20CM'
    labels[(freqs >= 2048) & (freqs < 4032) & ~arr] = '10CM'

    labels[(freqs < 960) & arr] = '40CM_uwl'
    labels[(freqs >= 960) & (freqs < 2048) & arr] = '20CM_uwl'
    labels[(freqs >= 2048) & (freqs < 4032) & arr] = '10CM_uwl'

    return labels

def selection_uwl_only(psr):
    backend_flags = np.asarray(psr.backend_flags, dtype=str)  
    labels = np.full_like(backend_flags, '', dtype=object)
    uwl_mask = np.char.find(backend_flags, 'UWL') >= 0
    labels[uwl_mask] = 'all_uwl'
    return labels

def selection_lowfreq(psr):

    return (np.asarray(psr.freqs) <= 960.0)

def selection_midfreq(psr):
    return (np.asarray(psr.freqs) > 960.0) & (np.asarray(psr.freqs) < 2048.0)

def selection_highfreq(psr):
    return (np.asarray(psr.freqs) >= 2048.0)

def gps2commongp(gps):
    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    
    ns = [gp.F.shape[1] for gp in gps]
    nmax = max(ns)
    
    def prior(params):
        yp = matrix.jnp.full((pmax, nmax), 1e-40)

        for i,p in enumerate(priors):
            yp = yp.at[i, :ns[i]].set(p(params))

        return yp
    prior.params = sorted(set([par for p in priors for par in p.params])) 
    
    Fs = [np.pad(gp.F, [(0,0), (0,nmax - gp.F.shape[1])]) for gp in gps]

    return matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)

get_red_comps = lambda psr: int(ds.getspan(psr) / 86400 / 240)
get_dm_comps = lambda psr: int(ds.getspan(psr) / 86400 / 60)
get_hf_comps = lambda psr: int(ds.getspan(psr) / 86400 / 30)


def make_solardm(psr, n_earth='constant'):


    theta, r_earth, _, _ = theta_impact(psr)
    shape = matrix.jnparray(AU_light_sec * AU_pc / r_earth / jnp.sinc(1 - theta/jnp.pi) * 4.148808e3 / psr.freqs**2)

    if n_earth == 'constant':
        def solardm():
            return shape * 4.0
    elif n_earth == 'uniform':
        def solardm(n_earth):
            return n_earth * shape

    return solardm

def make_swfourierbasis(psr, components, T=None):
    f, df, fmat = signals.fourierbasis(psr, components, T)
    theta, r_earth, _, _ = theta_impact(psr)
    shape = matrix.jnparray(AU_light_sec * AU_pc / r_earth / jnp.sinc(1 - theta/jnp.pi) * 4.148808e3 / psr.freqs**2)
    fmat *= shape[:, None]
    return f, df, fmat


def make_dm1yr(psr):

    fyr = 1.0 / sc.Julian_year
    def chrom_yearly_sinusoid(log10_Amp, phase):
        wf = 10**log10_Amp * jnp.sin(2 * jnp.pi * fyr * psr.toas + phase)
        return wf * (1400 / psr.freqs) ** 2
    return chrom_yearly_sinusoid

def make_dmgaussian(psr):

    def dm_gaussian(log10_Amp, epoch, log10_sigma):
        wf = 10**log10_Amp * jnp.exp(- (psr.toas - epoch*86400)**2 / (2* (10**log10_sigma * 86400)**2))
        return wf * (1400 / psr.freqs) ** 2
    return dm_gaussian

def make_chromefourierbasis(idx=4):
    def basis(psr, components, T=None):
        f, df, fmat = signals.fourierbasis(psr, components, T)
        Dm = (1400 / psr.freqs) ** idx
        return f, df, fmat * Dm[:, None]
    return basis

def sw_delay(psr, name='sw'):
    if psr.name in models_dict['sw']['uniform']:
        delay_sw = signals.makedelay(psr, make_solardm(psr, n_earth='uniform'), name=name)
    else:
        delay_sw = signals.makedelay(psr, make_solardm(psr, n_earth='constant'), name=name)
    return delay_sw

def dmexp_delay(psr, name='dmexp'):
    delay_dmexp_list = []
    if psr.name in models_dict['dm_exp'].keys():
        # print(f'PSR {psr.name} has {models_dict["dm_exp"][psr.name]} DM exponential dips.')
        num_dips = models_dict['dm_exp'][psr.name]
        for i in range(num_dips):
            idx = i + 1
            delay_dmexp = signals.makedelay(psr, make_chromaticdelay(psr), name=f'{name}_{idx}')
            delay_dmexp_list.append(delay_dmexp)
    return delay_dmexp_list

def dm1yr_delay(psr, name='dm1yr'):
    if psr.name in models_dict['dm1yr']:
        delay_dm1yr = signals.makedelay(psr, make_dm1yr(psr), name=name)
        return delay_dm1yr

def dmguassian_delay(psr, name='dmgauss'):

    if psr.name in models_dict['dmgauss']:
        delay_dmgauss = signals.makedelay(psr, make_dmgaussian(psr), name=name)
        return delay_dmgauss
    
def zero_gp(psr):
    F = jnp.zeros((psr.toas.shape[0], 0), dtype=jnp.float64)
    def prior(params):
        return matrix.jnp.array([], dtype=matrix.jnp.float64)
    prior.params = []
    return matrix.VariableGP(matrix.NoiseMatrix1D_var(prior), F)

def masked_fourierbasis(selection, base=signals.fourierbasis):
    def basis(psr, components, T=None):
        f, df, F = base(psr, components, T)
        m = selection(psr).astype(jnp.float64)           
        F = F * m[:, None]                              
        return f, df, F
    return basis

def red_gp(psr, name='rednoise'):
    gp_red = signals.makegp_fourier(psr, ds.powerlaw,
                                    components=get_red_comps(psr),
                                    name=name)
    return [gp_red]

def dm_gp(psr, name='dm_gp'):
    gp_dm = signals.makegp_fourier(psr, ds.powerlaw,
                                   components=get_dm_comps(psr),
                                   fourierbasis=signals.make_dmfourierbasis(alpha=2.0),
                                   name=name)
    return [gp_dm]

def sw_gp(psr, name='gp_sw'):
    if psr.name in models_dict['sw']['constant_with_gp']:
        gp_sw = signals.makegp_fourier(psr, ds.powerlaw,
                                       components=get_dm_comps(psr),
                                       fourierbasis=make_swfourierbasis, 
                                       name=name)
        return [gp_sw]
    else:
        return [zero_gp(psr)]


def chrom_gp(psr, name='chrom_gp'):
    if psr.name in models_dict['chrom']:
        gp_chrom = signals.makegp_fourier(psr, ds.powerlaw,
                                          components=get_red_comps(psr),
                                          fourierbasis=make_chromefourierbasis(idx=4),
                                          name=name)
        return [gp_chrom]
    else:
        return [zero_gp(psr)]

def bnlow_gp(psr, name='band_noise_low'):
    if psr.name in models_dict['bn_low']:
        gp_bnlow = signals.makegp_fourier(psr, ds.powerlaw,
                                          components=get_dm_comps(psr),
                                          fourierbasis=masked_fourierbasis(selection_lowfreq),
                                          name=name)
        return [gp_bnlow]
    else:
        return [zero_gp(psr)]

def bnmid_gp(psr, name='band_noise_mid'):
    if psr.name in models_dict['bn_mid']:
        gp_bnmid = signals.makegp_fourier(psr, ds.powerlaw,
                                          components=get_dm_comps(psr),
                                          fourierbasis=masked_fourierbasis(selection_midfreq),
                                          name=name)
        return [gp_bnmid]
    else:
        return [zero_gp(psr)]

def bnhigh_gp(psr, name='band_noise_high'):
    if psr.name in models_dict['bn_high']:
        gp_bnhigh = signals.makegp_fourier(psr, ds.powerlaw,
                                           components=get_dm_comps(psr),
                                           fourierbasis=masked_fourierbasis(selection_highfreq),
                                           name=name)
        return [gp_bnhigh]
    else:
        return [zero_gp(psr)]

def hf_gp(psr, name='hf_noise'):

    if psr.name in models_dict['hf']:
        gp_hf = signals.makegp_fourier(psr, ds.powerlaw,
                                       components=get_hf_comps(psr),
                                       name=name)
        return [gp_hf]
    else:
        return [zero_gp(psr)]

def get_cgp(psrs, gp_funcs):

    gps_per_psr = []
    for psr in psrs:
        lst = []
        for fn in gp_funcs:
            lst += fn(psr)        
        if not lst:
            lst = [zero_gp(psr)]  
        gps_per_psr.append(matrix.CompoundGP(lst))
    return gps2commongp(gps_per_psr)

def make_fourierbasis_ppta(logf=False, fmin=None, fmax=None, modes=None, pshift=False, pseed=None):

    def basis(psr, components, T=None):
        if T is None:
            T = signals.getspan(psr)

        if modes is not None:
            f = jnp.asarray(modes, dtype=jnp.float64)
            nmodes = len(f)
        else:
            if fmin is None: fmin_ = 1.0 / T
            else:            fmin_ = float(fmin)
            if fmax is None: fmax_ = components / T
            else:            fmax_ = float(fmax)

            nmodes = int(components)
            if logf:
                f = np.logspace(np.log10(fmin_), np.log10(fmax_), nmodes)
            else:
                f = np.linspace(fmin_, fmax_, nmodes)

        if pshift or (pseed is not None):
            seed = int(psr.toas[0] / 17.0) + int(pseed if pseed is not None else 0)
            rng = jnp.random.default_rng(seed)
            ranphase = rng.uniform(0.0, 2*jnp.pi, nmodes)
        else:
            ranphase = np.zeros(nmodes, dtype=np.float64)

        N = psr.toas.shape[0]
        F = np.zeros((N, 2*nmodes), dtype=np.float64)
        F[:, ::2] = np.sin(2.0 * np.pi * psr.toas[:, None] * f[None, :] + ranphase[None, :])
        F[:, 1::2] = np.cos(2.0 * np.pi * psr.toas[:, None] * f[None, :] + ranphase[None, :])

        df = np.diff(np.concatenate(([0.0], f)))
        return np.repeat(f, 2), np.repeat(df, 2), F
    return basis

def masked_fourierbasis_from_mask(mask, base=None):

    if base is None:
        base = signals.fourierbasis
    mask = jnp.asarray(mask).astype(jnp.float64)  
    def basis(psr, components, T=None):
        f, df, F = base(psr, components, T)
        return f, df, F * mask[:, None]
    return basis

def masks_by_group(psr, groups_attr='backend_flags', allow=None):

    g = np.asarray(getattr(psr, groups_attr))  
    vals = np.unique(g) if allow is None else np.unique(np.asarray(allow))
    out = {}
    for v in vals:
        m = (g == v)
        if m.any():
            out[str(v)] = m
    return out

def make_grouped_redgps(psr, components, T=None, group_noise=None,
                        groups_attr='backend_flags', name='red_group',
                        logf=False, fmin=None, fmax=None, modes=None,
                        pshift=False, pseed=None):

    base = make_fourierbasis_ppta(logf=logf, fmin=fmin, fmax=fmax, modes=modes,
                                  pshift=pshift, pseed=pseed)
    allow = None if group_noise is None else group_noise.get(psr.name, None)
    masks = masks_by_group(psr, groups_attr, allow=allow)

    gps = []
    for gname, mask in masks.items():
        fb = masked_fourierbasis_from_mask(mask, base=base)
        gp = signals.makegp_fourier(psr, ds.powerlaw, components=components,
                                    T=T, fourierbasis=fb, name=f'{name}_{gname}')
        gps.append(gp)
    return gps

def group_gp(psr, name='group_noises'):
    if psr.name in models_dict['group_noise']:
        group_noise = models_dict['group_noise']
        gps = make_grouped_redgps(psr, components=get_dm_comps(psr), group_noise=group_noise,
                                  groups_attr='backend_flags', name=name)
        return gps
    else:
        return [zero_gp(psr)]

def selection_group_ecorr(target_backend):
    def selection(psr):
        groups = np.asarray(psr.flags['group'], dtype=str)
        labels = np.full(groups.shape, '', dtype=object)
        mask = np.char.find(groups, str(target_backend)) >= 0
        labels[mask] = f'group_{target_backend}'
        return labels
    return selection

def makegp_ecorr_group(psr):
    ecorrs = []
    if psr.name in models_dict.get('group_ecorr', {}):
        for backend in models_dict['group_ecorr'][psr.name]:
            par = f"{psr.name}_group_{backend}_log10_ecorr"  
            if par in psr.noisedict:
                ecorrs.append(
                    signals.makegp_ecorr(
                        psr,
                        psr.noisedict,  
                        selection=selection_group_ecorr(backend),
                        name=f'ecorr_group_{backend}',
                    )
                )
    return ecorrs

def signal_gp(psr, common_gp=None):
    gps = []
    gps += red_gp(psr)  
    gps += dm_gp(psr)    
    gps += sw_gp(psr)    
    gps += chrom_gp(psr) 
    gps += bnlow_gp(psr) 
    gps += bnmid_gp(psr) 
    gps += bnhigh_gp(psr)
    gps += hf_gp(psr)    
    gps += group_gp(psr) 
    if common_gp is not None:
        gps += common_gp(psr)  
    return gps

def pptadr3(psrs, common_gp=None, crn=True, hd=False):
    pslmodels = [
                likelihood.PulsarLikelihood([
                    psr.residuals,
                    signals.makenoise_measurement(psr, psr.noisedict, tnequad=True),
                    signals.makegp_ecorr(psr, psr.noisedict, selection=selection_freqband, name='ecorr_band'),
                    signals.makegp_ecorr(psr, psr.noisedict, selection=selection_uwl_only, name='ecorr_uwl'),
                    *makegp_ecorr_group(psr),
                    signals.makegp_timing(psr, svd=True),
                    sw_delay(psr, name='sw'),
                    *dmexp_delay(psr, name='dmexp'),
                    dm1yr_delay(psr, name='dm1yr'),
                    dmguassian_delay(psr, name='dmgauss'),
                                            ])
                                            for psr in psrs
                ]
    cgp = gps2commongp([matrix.CompoundGP(signal_gp(psr, common_gp=common_gp)) for psr in psrs])
    if crn:
        corr = ds.hd_orf if hd else ds.uncorrelated_orf
        crn = signals.makeglobalgp_fourier(psrs, ds.powerlaw, corr, 
                                            components=get_red_comps(psrs),
                                            T=signals.getspan(psrs),
                                            name='crn')
    else:
        crn = None

    models = likelihood.ArrayLikelihood(pslmodels, 
                                        commongp=cgp,
                                        globalgp=crn
                                        )
    return models

if __name__ == "__main__":

    logx = ds.makelogtransform_uniform(models.logL, ppta_prior)
    num_samples = 128
    loss = dsf.value_and_grad_ElboLoss(logx, num_samples=num_samples)

    rng = jax.random.key(42)
    key, flow_key, train_key = jax.random.split(rng, 3)

    from flowjax.flows import triangular_spline_flow
    from flowjax.distributions import StandardNormal

    flow = triangular_spline_flow(flow_key,
                                base_dist=StandardNormal((len(logx.params),)), cond_dim=None,
                                flow_layers=16, knots=9, tanh_max_val=3.0, invert=False, init=None,)
    
    trainer = dsf.VariationalFit(dist=flow, loss_fn=loss, multibatch=8,
                             learning_rate=1e-2, annealing_schedule=lambda i: min(1.0, 0.5 + 0.5*i/500),
                             show_progress=True)
    
    train_key, trained_flow = trainer.run(train_key, steps=1001)

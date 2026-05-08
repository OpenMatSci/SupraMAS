#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mibo_with_llm_priors.py

Mechanism-Informed Bayesian Optimization prototype
- Reads history.xlsx (or change HISTORY_FILE) and optional mechanism.json
- Calls LLM via OpenAI(base_url=..., api_key=...) to parse mechanism entries (natural-language JSON)
- Maps LLM conclusions to parameter priors (mu, sigma for continuous; category weights for categorical)
- ParEGO-like scalarization + sklearn GPR surrogate + EI
- Candidate selection via EI * product(weights)
- Produces PDF report with LLM explanations and plots

Configure OPENAI_BASE_URL, OPENAI_API_KEY, and MIBO_MODEL_NAME in the environment when using the LLM prior parser.
"""

import os, json, math, random, re, warnings
from difflib import get_close_matches
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# -------------------- USER CONFIG: edit these --------------------
from openai import OpenAI
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MIBO_MODEL_NAME", "gpt-4o-mini")

HISTORY_FILE = "history.xlsx"   # Excel file with history data
MECH_FILE = "mechanism_structured.json"    # optional; may not exist
OUT_DIR = "mibo_output"
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------------------------------------------

# initialize client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY) if API_KEY else None

# -------------------- Robust LLM caller --------------------
def call_llm(prompt, temperature=0.2, max_tokens=512):
    """Call LLM via client.chat.completions.create and handle common response shapes."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot call the LLM prior parser.")
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role":"system","content":"As an expert in superconductor material processing and machine learning, your answers should be structured and practical."},
                {"role":"user","content":prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        # bubble up informative error
        raise RuntimeError(f"LLM call failed: {e}")

    # different sdk/bridge implementations return choices in slightly different shapes
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        pass
    try:
        return resp.choices[0].message["content"].strip()
    except Exception:
        pass
    try:
        return resp.choices[0].text.strip()
    except Exception:
        pass
    try:
        import json as _json
        return _json.dumps(resp, default=str, ensure_ascii=False)[:2000]
    except Exception:
        return str(resp)[:2000]

# -------------------- Utilities: fuzzy matching, polarity detection --------------------
def fuzzy_map_name(name, candidates, cutoff=0.6):
    if name is None:
        return None
    n = re.sub(r'[^\w\s]', ' ', name.lower()).strip()
    if not n:
        return None
    # substring
    for c in candidates:
        if n in c.lower().replace("_"," "):
            return c
    # difflib
    matches = get_close_matches(n, [c.lower() for c in candidates], n=1, cutoff=cutoff)
    if matches:
        m = matches[0]
        for c in candidates:
            if c.lower() == m:
                return c
    return None

_POSITIVE_PATTERNS = [r'\benhance\b', r'\bincrease\b', r'\bpromote\b', r'\bimprove\b', r'\bboost\b', r'\bpositive\b', r'\benhances\b', r'\bmore\b']
_NEGATIVE_PATTERNS = [r'\breduce\b', r'\bdecrease\b', r'\bsuppress\b', r'\binhibit\b', r'\bnegative\b', r'\blower\b']

def detect_polarity(text):
    if not text:
        return 0
    t = text.lower()
    for p in _POSITIVE_PATTERNS:
        if re.search(p, t):
            return +1
    for p in _NEGATIVE_PATTERNS:
        if re.search(p, t):
            return -1
    return 0

# -------------------- Parse LLM mechanistic entries -> structured priors --------------------
def parse_llm_mechanisms(llm_output, param_candidates, target_candidates=None):
    """
    Accept LLM output (string or parsed JSON). Expected item keys:
      e.g. {"label_factor":"grain size","factor":"Radius ...","label_outcome":"grain growth","outcome":"...","response":"enhances growth of larger grains","mechanism":"..."}
    Returns dict mapping matched_param -> entry dict: polarity, suggested_interval/list, confidence, comment, matched_outcome, raw.
    """
    # ensure we have parsed JSON if possible
    entries = []
    parsed = None
    if isinstance(llm_output, str):
        # try to parse JSON out of it
        try:
            parsed = json.loads(llm_output)
        except Exception:
            # try to extract JSON-like block from text
            m = re.search(r'\{[\s\S]*\}', llm_output)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
    else:
        parsed = llm_output

    if parsed is None:
        # cannot parse structured JSON; return empty
        return {}

    if isinstance(parsed, dict):
        # either single entry dict or mapping param->dict; detect
        candidate_keys = set(parsed.keys())
        # heuristic: if keys include label_factor or factor -> treat as single entry
        if 'factor' in candidate_keys or 'label_factor' in candidate_keys or 'response' in candidate_keys:
            entries = [parsed]
        else:
            # treat as mapping param->spec
            for k,v in parsed.items():
                if isinstance(v, dict):
                    e = v.copy()
                    if 'factor' not in e:
                        e['factor'] = k
                    entries.append(e)
    elif isinstance(parsed, list):
        entries = parsed
    else:
        return {}

    result = {}
    for e in entries:
        factor = e.get('factor') or e.get('label_factor') or e.get('label') or None
        outcome = e.get('outcome') or e.get('label_outcome') or None
        response = e.get('response') or e.get('effect') or ""
        mech_text = e.get('mechanism') or e.get('comment') or ""
        suggested_interval = e.get('interval') or e.get('optimal_range') or None
        suggested_list = e.get('list') or e.get('preferred') or None
        confidence = e.get('confidence', None)

        matched_param = fuzzy_map_name(factor or "", param_candidates)
        matched_outcome = fuzzy_map_name(outcome or "", target_candidates) if target_candidates else None
        polarity = detect_polarity((response or "") + " " + (mech_text or ""))

        entry = {
            'raw': e,
            'factor': factor,
            'outcome': outcome,
            'matched_param': matched_param,
            'matched_outcome': matched_outcome,
            'polarity': polarity,
            'suggested_interval': suggested_interval,
            'suggested_list': suggested_list,
            'confidence': float(confidence) if confidence is not None else None,
            'comment': mech_text,
            'response': response
        }
        key = matched_param if matched_param else f"_unmapped_{len(result)+1}"
        result[key] = entry
    return result

# -------------------- Convert LLM priors + history stats -> numeric priors --------------------
def continuous_prior_from_polarity(hist_stats, polarity, mech_hint=None):
    """
    hist_stats: dict with mean,std,min,max (numbers or None)
    Return mu,sigma (floats) or (None,None) if insufficient data
    """
    if mech_hint and isinstance(mech_hint, dict):
        iv = mech_hint.get('suggested_interval') or mech_hint.get('interval') or mech_hint.get('optimal_range')
        if iv and isinstance(iv, (list,tuple)) and len(iv)==2:
            a,b = float(iv[0]), float(iv[1])
            mu = 0.5*(a+b)
            sigma = max((b-a)/4.0, 1e-6)
            return mu, sigma
    mean = hist_stats.get('mean')
    std = hist_stats.get('std')
    lo = hist_stats.get('min')
    hi = hist_stats.get('max')
    if mean is not None and std is not None and lo is not None and hi is not None:
        if polarity > 0:
            mu = min(mean + 0.6*std, hi)
        elif polarity < 0:
            mu = max(mean - 0.6*std, lo)
        else:
            mu = mean
        sigma = max(std, (hi-lo)/12.0)
        return float(mu), float(sigma)
    if lo is not None and hi is not None:
        mu = 0.75*hi + 0.25*lo if polarity>0 else (0.25*hi + 0.75*lo if polarity<0 else 0.5*(lo+hi))
        sigma = max((hi-lo)/6.0, 1e-6)
        return float(mu), float(sigma)
    return None, None

def categorical_prior_from_llm(suggested_list, hist_freqs):
    # hist_freqs: dict cat->freq
    if not hist_freqs:
        return {}
    total = sum(hist_freqs.values()) if sum(hist_freqs.values())>0 else 1.0
    base = {c: hist_freqs.get(c,0.0)/total for c in hist_freqs}
    weights = {}
    for c in base:
        w = 0.3 + 0.7 * base[c]
        if suggested_list and c in suggested_list:
            w = min(1.0, w * 1.3)
        weights[c] = w
    return weights

def build_resolved_priors(llm_priors, hist_stats_map, cat_freqs_map):
    resolved = {}
    for key,entry in llm_priors.items():
        p = entry.get('matched_param')
        if p and p in hist_stats_map:
            mu_sigma = continuous_prior_from_polarity(hist_stats_map[p], entry.get('polarity',0), mech_hint=entry)
            if mu_sigma[0] is None:
                mu_sigma = (hist_stats_map[p].get('mean'), hist_stats_map[p].get('std') or ((hist_stats_map[p].get('max',1)-hist_stats_map[p].get('min',0))/6.0))
            resolved[p] = {'mu': float(mu_sigma[0]), 'sigma': float(max(mu_sigma[1],1e-6)), 'raw': entry}
        elif p:
            # categorical
            freqs = cat_freqs_map.get(p, {})
            cat_weights = categorical_prior_from_llm(entry.get('suggested_list') or [], freqs)
            resolved[p] = {'cat_weights': cat_weights, 'raw': entry}
        else:
            resolved[key] = {'raw': entry}
    return resolved

# -------------------- Main MIBO flow --------------------
def run_mibo():
    import os, json, math, random, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.exceptions import ConvergenceWarning
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    warnings.filterwarnings("ignore")

    # ------------------ Config ------------------
    NUM_CANDIDATES = 3
    OUT_DIR = "mibo_output"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------ Load history ------------------
    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(f"{HISTORY_FILE} not found")
    hist_df = pd.read_excel(HISTORY_FILE)
    hist_df.columns = hist_df.columns.str.strip()

    desired_params = ["deposition Temperature","processing atmosphere","annealing Temperature","annealing conditions"]
    desired_targets = ["77K,0T","4.2K,7T"]

    # fuzzy resolve col names
    def fuzzy_find(cols, desired):
        out=[]
        cols_list = list(cols)
        for d in desired:
            found=None
            dl = d.lower().replace(" ","").replace("_","").replace("-","")
            for c in cols_list:
                cl = c.lower().replace(" ","").replace("_","").replace("-","")
                if dl in cl or cl in dl:
                    found=c; break
            if found is None:
                for c in cols_list:
                    if any(part in c.lower() for part in d.lower().split()):
                        found=c; break
            out.append(found)
        return out

    param_cols = fuzzy_find(hist_df.columns, desired_params)
    target_cols = fuzzy_find(hist_df.columns, desired_targets)
    for i,p in enumerate(param_cols):
        if p is None: raise ValueError(f"Cannot find parameter column matching '{desired_params[i]}'.")
    for i,t in enumerate(target_cols):
        if t is None: raise ValueError(f"Cannot find target column matching '{desired_targets[i]}'.")

    print("Parameter cols:", param_cols)
    print("Target cols:", target_cols)

    # ------------------ Build X/Y ------------------
    X_df = hist_df[param_cols].copy()
    Y_df = hist_df[target_cols].copy()

    # detect continuous vs categorical robustly
    cont_cols, cat_cols = [], []
    for c in X_df.columns:
        coerced = pd.to_numeric(X_df[c], errors='coerce')
        if coerced.notna().sum() >= max(1, int(0.5*len(X_df))):
            cont_cols.append(c); X_df[c] = coerced
        else:
            cat_cols.append(c)

    n_rows = len(X_df)
    X_cont = X_df[cont_cols].astype(float).fillna(method='ffill').values if cont_cols else np.zeros((n_rows,0))
    if X_cont.ndim == 1: X_cont = X_cont.reshape(n_rows,1)
    if cat_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        ohe.fit(X_df[cat_cols].astype(str).values)
        X_cat = ohe.transform(X_df[cat_cols].astype(str).values)
        if X_cat.ndim == 1: X_cat = X_cat.reshape(n_rows,1)
    else:
        ohe = None
        X_cat = np.zeros((n_rows,0))
    X_all = np.hstack([X_cont, X_cat]).astype(float)
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X_all)

    Y_all = Y_df.astype(float).fillna(method='ffill').values
    n_samples, n_targets = Y_all.shape

    # ------------------ Load mechanism JSON (optional) ------------------
    mech_data = None
    if os.path.exists(MECH_FILE):
        try:
            with open(MECH_FILE, "r", encoding="utf-8") as f:
                mech_data = json.load(f)
        except Exception:
            with open(MECH_FILE, "r", encoding="latin1") as f:
                mech_data = json.load(f)

    # ------------------ History stats ------------------
    param_stats = {}
    cat_freqs_map = {}
    for p in param_cols:
        arr = pd.to_numeric(hist_df[p], errors='coerce').dropna()
        param_stats[p] = {
            "mean": float(arr.mean()) if len(arr)>0 else None,
            "std": float(arr.std()) if len(arr)>0 else None,
            "min": float(arr.min()) if len(arr)>0 else None,
            "max": float(arr.max()) if len(arr)>0 else None,
        }
        if p in cat_cols:
            cat_freqs_map[p] = hist_df[p].astype(str).value_counts(normalize=True).to_dict()

    # ------------------ Ask LLM for mechanisms -> priors ------------------
    resolved_priors = {}
    llm_parsed = {}
    if mech_data:
        prompt = (
            "Please analyze the following mechanism entries and parameter statistics, and return a JSON mapping "
            "each parameter to either {interval:[a,b], confidence:0..1, comment:'...'} for continuous parameters, "
            "or {list:['cat1','cat2'], confidence:0..1, comment:'...'} for categorical parameters. "
            "Do not add extra text, only valid JSON.\n\n"
            "Mechanism entries:\n" + json.dumps(mech_data, ensure_ascii=False, indent=2) +
            "\n\nParameter statistics:\n" + json.dumps(param_stats, ensure_ascii=False, indent=2)
        )
        try:
            llm_out = call_llm(prompt, temperature=0.0, max_tokens=900)
        except Exception as e:
            llm_out = f"(LLM call failed: {e})"
        # parse into structured entries aligned to parameters
        llm_parsed = parse_llm_mechanisms(llm_out, param_candidates=param_cols, target_candidates=target_cols)
        resolved_priors = build_resolved_priors(llm_parsed, param_stats, cat_freqs_map)
    else:
        resolved_priors = {}

    # ------------------ Collect raw mechanistic evidence per parameter ------------------
    # We extract original sentences from llm_parsed[*]['raw'] fields (keys like 'mechanism','response','outcome', etc.)
    mech_evidence = {p: [] for p in param_cols}
    for key, entry in (llm_parsed or {}).items():
        p = entry.get('matched_param')
        if not p: continue
        raw = entry.get('raw', {})
        # concatenate plausible evidence fields
        snips = []
        for kk in ['mechanism', 'response', 'outcome', 'factor', 'comment']:
            vv = raw.get(kk)
            if isinstance(vv, str) and vv.strip():
                snips.append(vv.strip())
        if snips:
            mech_evidence.setdefault(p, []).append(" ".join(snips))

    # ------------------ Fuse priors: mu/sigma for continuous; weights for categorical ------------------
    final_mu_sigma = {}
    for p in param_cols:
        if p in cont_cols:
            hist = param_stats.get(p, {})
            mu = hist.get('mean')
            if mu is None and (hist.get('min') is not None and hist.get('max') is not None):
                mu = 0.5*(hist['min']+hist['max'])
            sigma = hist.get('std')
            if sigma is None and (hist.get('min') is not None and hist.get('max') is not None):
                sigma = (hist['max']-hist['min'])/6.0
            if sigma is None: sigma = 1.0
            ll = resolved_priors.get(p)
            if ll and 'mu' in ll:
                mu_ll, sigma_ll = ll['mu'], ll['sigma']
                conf = ll.get('raw', {}).get('confidence')
                w = float(conf) if conf is not None else 0.5
                mu = w*mu_ll + (1-w)*mu
                sigma = max(w*sigma_ll + (1-w)*sigma, 1e-6)
            final_mu_sigma[p] = (float(mu), float(sigma))
        else:
            ll = resolved_priors.get(p)
            if ll and 'cat_weights' in ll and ll['cat_weights']:
                final_mu_sigma[p] = ('categorical', ll['cat_weights'])
            else:
                vc = cat_freqs_map.get(p, {})
                weights = {c: 0.3 + 0.7*vc.get(c,0.0) for c in vc}
                final_mu_sigma[p] = ('categorical', weights)

    print("\nResolved priors (from LLM + history):")
    for k, v in final_mu_sigma.items():
        print(" ", k, "->", v)

    # ------------------ Train scalarized surrogate (ParEGO-style Tchebycheff) ------------------
    Y_np = np.array(Y_all)
    ideal = np.min(-Y_np, axis=0)
    w_raw = np.random.exponential(1.0, size=Y_np.shape[1])
    w = w_raw / w_raw.sum()
    scalar_vals = np.array([np.max(w*np.abs(-Y_np[i]-ideal)) for i in range(Y_np.shape[0])])
    scalar_for_model = -scalar_vals

    kernel = ConstantKernel(1.0)*Matern(length_scale=np.ones(X_scaled.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-6)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X_scaled, scalar_for_model)

    # ------------------ Sampling & scoring (EI * prior weights) ------------------
    def sample_candidates(num=3000):
        samples=[]
        for _ in range(num):
            s={}
            for p in param_cols:
                if p in cont_cols:
                    arr = pd.to_numeric(hist_df[p], errors='coerce').dropna()
                    if len(arr)>0:
                        lo, hi = float(arr.min()), float(arr.max())
                        if lo == hi:
                            span = max(abs(lo)*0.1, 1.0); lo -= span; hi += span
                    else:
                        lo, hi = 0.0, 1.0
                    s[p] = float(np.random.uniform(lo, hi))
                else:
                    vals = hist_df[p].astype(str).dropna().unique().tolist()
                    s[p] = random.choice(vals) if len(vals)>0 else ""
            samples.append(s)
        return samples

    def sample_to_row_scaled(sample):
        cont_vals = [float(sample[c]) for c in cont_cols] if cont_cols else []
        cat_vals = [str(sample[c]) for c in cat_cols] if cat_cols else []
        cat_enc = ohe.transform([cat_vals])[0] if cat_cols else np.zeros((0,))
        row = np.hstack([cont_vals, cat_enc]).astype(float)
        return row, x_scaler.transform([row])[0]

    def weight_cont(x, mu, sigma, min_explore=0.3):
        wv = math.exp(-0.5*((x-mu)/(sigma+1e-12))**2)
        return min_explore + (1-min_explore)*wv

    candidates = sample_candidates(num=3000)
    f_best = np.max(scalar_for_model)
    scored = []
    for cand in candidates:
        try:
            _, x_scaled_row = sample_to_row_scaled(cand)
        except Exception:
            continue
        mu_pred, sigma_pred = gpr.predict([x_scaled_row], return_std=True)
        mu_pred, sigma_pred = float(mu_pred[0]), float(sigma_pred[0])
        if sigma_pred <= 1e-9:
            ei = 0.0
        else:
            z = (mu_pred - f_best) / sigma_pred
            ei = max((mu_pred - f_best)*norm.cdf(z) + sigma_pred*norm.pdf(z), 0.0)
        wprod = 1.0
        for p in param_cols:
            if p in cont_cols:
                mu_p, sigma_p = final_mu_sigma[p]
                wprod *= weight_cont(float(cand[p]), mu_p, sigma_p)
            else:
                tag, weights_map = final_mu_sigma[p]
                wprod *= (weights_map.get(str(cand[p]), 0.3) if isinstance(weights_map, dict) else 0.3)
        scored.append((ei*wprod, cand.copy(), {"ei":ei,"wprod":wprod,"mu_pred":mu_pred,"sigma_pred":sigma_pred}))

    best_candidates = sorted(scored, key=lambda x: x[0], reverse=True)[:NUM_CANDIDATES]

    # ------------------ Per-objective GPRs for reporting ------------------
    per_gpr = []
    for j in range(n_targets):
        g = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            g.fit(X_scaled, Y_all[:, j])
        per_gpr.append(g)

    # ------------------ Build PDF report (English) ------------------
    doc = SimpleDocTemplate(os.path.join(OUT_DIR, "mibo_report.pdf"))
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("MIBO Report (LLM-assisted; per-candidate explanations with mechanistic evidence)", styles['Title']))
    story.append(Spacer(1, 8))

    # Show resolved priors up-front for transparency
    story.append(Paragraph("Resolved Priors (from LLM + History):", styles['Heading2']))
    for p, v in final_mu_sigma.items():
        if isinstance(v, tuple):
            story.append(Paragraph(f"{p}: mu = {v[0]:.4f}, sigma = {v[1]:.4f}", styles['Normal']))
        else:
            _, weights_map = v
            weights_str = ", ".join([f"{k}:{w:.2f}" for k, w in weights_map.items()])
            story.append(Paragraph(f"{p}: categorical weights -> {weights_str}", styles['Normal']))
    story.append(Spacer(1, 10))

    preds_per_candidate = []
    for idx, (score, cand, meta) in enumerate(best_candidates):
        story.append(Paragraph(f"Candidate {idx+1}", styles['Heading2']))

        # Parameters table
        tbl_params = [["Parameter", "Value"]] + [[k, str(v)] for k, v in cand.items()]
        story.append(Table(tbl_params, colWidths=[220, 220],
                           style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.black),
                                             ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)])))
        story.append(Spacer(1, 4))

        # Predictions (per objective using per-objective GPR)
        _, cand_scaled = sample_to_row_scaled(cand)
        preds = [float(g.predict([cand_scaled])[0]) for g in per_gpr]
        preds_per_candidate.append(preds)
        tbl_preds = [["Objective", "Predicted value"]] + [[target_cols[i], f"{preds[i]:.6f}"] for i in range(n_targets)]
        story.append(Table(tbl_preds, colWidths=[220, 220],
                           style=TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.black),
                                             ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)])))
        story.append(Spacer(1, 4))

        # BO meta
        story.append(Paragraph(
            f"Bayesian Optimization Meta: EI = {meta['ei']:.6f}; Weight product = {meta['wprod']:.3f}; "
            f"GPR mean = {meta['mu_pred']:.6f}; std = {meta['sigma_pred']:.6f}",
            styles['Normal']
        ))
        story.append(Spacer(1, 4))

        # Prediction bar chart
        plt.figure(figsize=(5, 2.6))
        plt.bar(range(n_targets), preds)
        plt.xticks(range(n_targets), target_cols, rotation=0)
        plt.ylabel("Predicted")
        plt.title(f"Candidate {idx+1} Predictions")
        img_path = os.path.join(OUT_DIR, f"candidate_{idx+1}_preds.png")
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        story.append(Image(img_path, width=420, height=200))
        story.append(Spacer(1, 4))

        # Mechanistic evidence used (original sentences/snippets)
        story.append(Paragraph("Mechanistic evidence used:", styles['Heading3']))
        any_evidence = False
        for p in param_cols:
            ev_list = mech_evidence.get(p, [])
            if not ev_list: continue
            any_evidence = True
            story.append(Paragraph(f"- {p}:", styles['Normal']))
            # show up to 3 snippets per parameter
            for s_idx, snip in enumerate(ev_list[:3], 1):
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;• {snip}", styles['Normal']))
        if not any_evidence:
            story.append(Paragraph("No explicit mechanistic snippets available for this candidate.", styles['Normal']))
        story.append(Spacer(1, 6))

        # LLM Decision Explanation (English, referencing which mechanisms influenced selection)
        expl_payload = {
            "candidate": cand,
            "predicted_objectives": dict(zip(target_cols, preds)),
            "meta": meta,
            "resolved_priors": resolved_priors,
            "mechanistic_snippets_by_param": {k: v[:3] for k, v in mech_evidence.items() if v},
            "note": "Explain which specific mechanism snippets (above) influenced each parameter choice and how EI*weights led to selection."
        }
        try:
            expl_text = call_llm(
                "Provide a 5–8 sentence English rationale for selecting this candidate. "
                "Explicitly refer to the listed mechanistic snippets (by quoting short phrases) and describe how they shape the prior-weighted acquisition (EI × weights). "
                "Also mention any trade-offs across the three objectives and recommended next validation steps.\n\n"
                + json.dumps(expl_payload, ensure_ascii=False, indent=2),
                temperature=0.2, max_tokens=450
            )
        except Exception as e:
            expl_text = f"(LLM explanation failed: {e})"

        story.append(Paragraph("LLM Decision Explanation:", styles['Heading3']))
        story.append(Paragraph(expl_text, styles['Normal']))
        story.append(Spacer(1, 12))

    # Footer: show resolved priors dictionary (for full transparency)
    story.append(Paragraph("Resolved Priors from LLM + History (JSON):", styles['Heading2']))
    story.append(Paragraph(json.dumps(resolved_priors, ensure_ascii=False, indent=2)[:4000], styles['Code']))
    story.append(Spacer(1, 6))

    # Build PDF
    doc.build(story)
    print(f"\nReport saved: {os.path.join(OUT_DIR, 'mibo_report.pdf')}")
if __name__ == "__main__":
    run_mibo()

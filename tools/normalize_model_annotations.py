"""Normalize model_annotations/ into <case>/<model_id>[__<variant>]/<witness>.csv.

Run with --dry-run (default) to preview; pass --apply to perform `git mv`.
Model ids were verified against the data (the 'thinking' column fingerprints
R1-distill / reasoning models). The alias 'qwen7b' flips meaning by case, so it
is resolved per-directory from that signal.
"""
import argparse, glob, subprocess, sys
import numpy as np, pandas as pd
from pathlib import Path

NORM_CASE = {'set2_enron_d':'enron_d','set2_enron_p':'enron_p',
             'set3_simpson_d':'simpson_d','set3_simpson_p':'simpson_p',
             'WMT_D':'WMT_D','WMT_P':'WMT_P'}

# alias -> (model_id, variant)   [variant '' = default prompt]
ALIAS = {
 '4o_mini':('gpt-4o-mini',''),'GPT4o_mini':('gpt-4o-mini',''),'gpt4omini':('gpt-4o-mini',''),
 'o1_mini':('o1-mini',''),'o3_mini':('o3-mini',''),
 'gemini_non_re':('gemini-flash',''),'gemini_non_reasoning':('gemini-flash',''),'Gemini_Flash_OFF':('gemini-flash',''),
 'gemini_re':('gemini-flash-thinking',''),'gemini_reasoning':('gemini-flash-thinking',''),'Gemini_Flash_ON':('gemini-flash-thinking',''),
 'llaman8b':('llama-3.1-8b',''),'Llama3.1_8B':('llama-3.1-8b',''),'llama8b_n':('llama-3.1-8b',''),
 'llama8b':('r1-distill-llama-8b',''),'DS_Llama_8B':('r1-distill-llama-8b',''),'deepseek8b':('r1-distill-llama-8b',''),
 'llaman_70b':('llama-3.3-70b',''),'Llama3.3_70B':('llama-3.3-70b',''),'llama70b_normal':('llama-3.3-70b',''),
 'llama_r_70b':('r1-distill-llama-70b',''),'DS_Llama_70B':('r1-distill-llama-70b',''),'r1_llama_70b':('r1-distill-llama-70b',''),
 'qwen_n_7b':('qwen2.5-7b',''),'qwen_n7b':('qwen2.5-7b',''),'qwen7b_n':('qwen2.5-7b',''),'Qwen_7B':('qwen2.5-7b',''),
 'qwen7':('r1-distill-qwen-7b',''),'qwen_r_7b':('r1-distill-qwen-7b',''),'DS_Qwen_7B':('r1-distill-qwen-7b',''),'r1_qwen_7b':('r1-distill-qwen-7b',''),
 'qwen25':('qwen2.5-32b',''),'qwen32':('qwen2.5-32b',''),'Qwen_32B':('qwen2.5-32b',''),
 'qwq':('qwq-32b',''),'QwQ_32B':('qwq-32b',''),
 'qwen_32_cons':('qwen2.5-32b','constitution'),'qwen32_cons':('qwen2.5-32b','constitution'),'qwen32_cons_shot':('qwen2.5-32b','constitution'),
 'qwen_32_few':('qwen2.5-32b','few-shot'),'qwen32_few':('qwen2.5-32b','few-shot'),'qwen32_few_shot':('qwen2.5-32b','few-shot'),
 'qwq_32_cons':('qwq-32b','constitution'),'qwq_cons':('qwq-32b','constitution'),
 'qwq_32_few':('qwq-32b','few-shot'),'qwq_few':('qwq-32b','few-shot'),
}
WMT_WITNESS = {'WMT_D':'JM_ofshe','WMT_P':'JM_detective'}

def thinking_frac(files):
    v=[]
    for f in files:
        df=pd.read_csv(f)
        t=df['thinking'].astype(str) if 'thinking' in df.columns else pd.Series([],dtype=str)
        v.append(((t.str.len()>5)&(t!='nan')).mean() if len(t) else 0.0)
    return float(np.mean(v)) if v else 0.0

def flat_alias(stem):
    s=stem.replace('JM_ofshe_','').replace('JM_detective_','')
    s=s.replace('annotated_','').replace('_annotated','')
    return s

def resolve(alias, files):
    if alias=='qwen7b':                       # meaning flips by case
        return ('r1-distill-qwen-7b','') if thinking_frac(files)>0.5 else ('qwen2.5-7b','')
    return ALIAS.get(alias)

def plan():
    moves=[]; unknown=[]
    # subdir cases
    for case in ['set2_enron_d','set2_enron_p','set3_simpson_d','set3_simpson_p']:
        for d in sorted(glob.glob(f'model_annotations/{case}/*/')):
            alias=Path(d).name; files=sorted(glob.glob(d+'*.csv'))
            r=resolve(alias,files)
            if not r: unknown.append((case,alias)); continue
            mid,var=r; sub=mid+(f'__{var}' if var else '')
            for f in files:
                tgt=f'model_annotations/{NORM_CASE[case]}/{sub}/{Path(f).name}'
                moves.append((f,tgt))
    # flat cases
    for case,subdir in [('WMT_D','LLM_annotated'),('WMT_D','Qwen_extra'),('WMT_P','')]:
        base=f'model_annotations/{case}/{subdir}'.rstrip('/')
        for f in sorted(glob.glob(base+'/*.csv')):
            alias=flat_alias(Path(f).stem); r=resolve(alias,[f])
            if not r: unknown.append((case,alias)); continue
            mid,var=r; sub=mid+(f'__{var}' if var else '')
            tgt=f'model_annotations/{NORM_CASE[case]}/{sub}/{WMT_WITNESS[case]}.csv'
            moves.append((f,tgt))
    return moves, unknown

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--apply',action='store_true'); a=ap.parse_args()
    moves,unknown=plan()
    tgts=[t for _,t in moves]
    collisions={t for t in tgts if tgts.count(t)>1}
    print(f"planned moves: {len(moves)} | unknown aliases: {len(unknown)} | collisions: {len(collisions)}")
    if unknown: print("UNKNOWN:",unknown)
    if collisions:
        print("COLLISIONS:"); [print("  ",c) for c in sorted(collisions)]
    # show the new structure (case -> model dirs)
    from collections import defaultdict
    tree=defaultdict(set)
    for _,t in moves:
        p=Path(t).parts; tree[p[1]].add(p[2])
    for case in sorted(tree): print(f"\n{case}/  ({len(tree[case])} model dirs)\n  "+"  ".join(sorted(tree[case])))
    if a.apply and not unknown and not collisions:
        for s,t in moves:
            Path(t).parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','mv',s,t],check=True)
        print(f"\nAPPLIED {len(moves)} git mv operations.")
    elif a.apply:
        print("\nNOT applied: resolve unknown/collisions first."); sys.exit(1)

if __name__=='__main__': main()

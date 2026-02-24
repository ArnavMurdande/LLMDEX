import json
import os
import argparse
from typing import Dict, List, Any

def detect_sub_family(name: str, provider: str) -> str:
    name_lower = name.lower()
    if 'claude' in name_lower:
        if 'opus' in name_lower: return 'Claude Opus'
        if 'sonnet' in name_lower: return 'Claude Sonnet'
        if 'haiku' in name_lower: return 'Claude Haiku'
        return 'Claude Base'
    if 'gemini' in name_lower:
        if 'pro' in name_lower: return 'Gemini Pro'
        if 'flash' in name_lower: return 'Gemini Flash'
        if 'ultra' in name_lower: return 'Gemini Ultra'
        if 'nano' in name_lower: return 'Gemini Nano'
        return 'Gemini Base'
    if 'gpt' in name_lower:
        if 'mini' in name_lower: return 'GPT Mini'
        if 'medium' in name_lower: return 'GPT Medium'
        if 'high' in name_lower: return 'GPT High'
        if 'codex' in name_lower: return 'GPT Codex'
        if 'pro' in name_lower: return 'GPT Pro'
        if 'o1' in name_lower or 'o3' in name_lower or 'o4' in name_lower:
            return 'OpenAI o-series'
        return 'GPT Base/Flagship'
    if 'deepseek' in name_lower:
        if 'r1' in name_lower: return 'DeepSeek R'
        if 'v' in name_lower or 'coder' in name_lower: return 'DeepSeek V/Coder'
        return 'DeepSeek Base'
    if 'qwen' in name_lower:
        if 'max' in name_lower: return 'Qwen Max'
        if 'plus' in name_lower: return 'Qwen Plus'
        if 'turbo' in name_lower: return 'Qwen Turbo'
        if 'math' in name_lower or 'coder' in name_lower: return 'Qwen Specialty'
        return 'Qwen Base'
    if 'llama' in name_lower:
        return 'Meta Llama'
    if 'mistral' in name_lower or 'mixtral' in name_lower:
        return 'Mistral AI'
    
    # default to provider if known, else split first word
    if provider and provider.lower() not in ['unknown', 'other']:
        return provider
    return name.split()[0] if name else 'Unknown'

def build_history():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    latest_path = os.path.join(base_dir, "index", "latest.json")
    
    if not os.path.exists(latest_path):
        print(f"File not found: {latest_path}")
        return
        
    with open(latest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    families: Dict[str, List[Any]] = {}
    
    for row in data:
        name = row.get("canonical_name") or row.get("model_name")
        if not name:
            continue
            
        # Use existing model_family or fallback to regex detection
        fam = row.get("model_family")
        if not fam:
            fam = detect_sub_family(name, row.get("provider"))
            
        perf = row.get("adjusted_performance")
        if perf is None:
            perf = row.get("performance_index")
            
        if perf is None:
            continue
            
        if fam not in families:
            families[fam] = []
            
        families[fam].append({
            "name": name,
            "performance": perf,
            "rank": row.get("performance_rank"),
            "provider": row.get("provider"),
            "family_order": row.get("family_order")
        })
        
    # Sort and compute improvements
    history_output = {}
    
    for fam, members in families.items():
        # First sort by family_order if available, then by performance ascending (assuming newer models perform better)
        # We group members that have the same order by taking the best performing one
        members.sort(key=lambda x: (x.get("family_order") or 999, x["performance"]))
        
        # We assume progression is performance-based if family_order is identical or missing
        # So low to high performance represents progression.
        for i, m in enumerate(members):
            if i == 0:
                m["improvement_abs"] = 0.0
                m["improvement_pct"] = 0.0
                m["predecessor"] = None
            else:
                prev = members[i-1]
                diff = m["performance"] - prev["performance"]
                pct_diff = diff / prev["performance"] * 100 if prev["performance"] else 0
                
                m["improvement_abs"] = round(diff, 2)
                m["improvement_pct"] = round(pct_diff, 2)
                m["predecessor"] = prev["name"]
                
        history_output[fam] = members
        
    # Drop families with only 1 member
    history_output = {k: v for k, v in history_output.items() if len(v) > 1}
    
    history_dir = os.path.join(base_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    out_path = os.path.join(history_dir, "family_growth.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history_output, f, indent=2)
        
    print(f"Model family history successfully built with {len(history_output)} families!")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    build_history()

import json
import glob
import statistics
import os

def analyze_jsonl(file_path):
    input_tokens = []
    output_tokens = []
    thinking_tokens = []
    
    with open(file_path, "r") as f:
        for line in f:
            if "response" in line or "response\"" in line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        resp = data["response"]
                        usage = resp.get("usageMetadata", resp.get("usage", {}))
                        
                        if usage:
                            i_tok = usage.get("promptTokenCount")
                            o_tok = usage.get("candidatesTokenCount")
                            t_tok = usage.get("thoughtsTokenCount", 0) 
                            
                            if i_tok is not None:
                                input_tokens.append(i_tok)
                            if o_tok is not None:
                                output_tokens.append(o_tok)
                            if t_tok is not None:
                                thinking_tokens.append(t_tok)
                except Exception:
                    continue

    if not input_tokens:
        return None
        
    return {
        "file": os.path.basename(file_path),
        "count": len(input_tokens),
        "mean_input": statistics.mean(input_tokens),
        "mean_output": statistics.mean(output_tokens) if output_tokens else 0,
        "mean_thinking": statistics.mean(thinking_tokens) if thinking_tokens else 0,
        "max_thinking": max(thinking_tokens) if thinking_tokens else 0,
    }

def main():
    xland_files = glob.glob("batch_outputs_difficulty_feedback_prediction*.jsonl")
    jaxnav_files = glob.glob("jaxnav_batch_outputs_prediction*.jsonl")
    
    if not xland_files:
        print("Could not find XLand output JSONL.")
    else:
        print(f"Analyzing XLand File: {xland_files[0]}...")
        x_stats = analyze_jsonl(xland_files[0])
        if x_stats:
            print("\n=== XLand-MiniGrid Statistics ===")
            print(f"Total Processed Pairs: {x_stats['count']}")
            print(f"Mean Input Tokens:     {x_stats['mean_input']:.1f}")
            print(f"Mean Output Tokens:    {x_stats['mean_output']:.1f} (Includes text + thinking)")
            print(f"Mean Thinking Tokens:  {x_stats['mean_thinking']:.1f}")
            print(f"Max Thinking Tokens:   {x_stats['max_thinking']}")
            
    if not jaxnav_files:
        print("\nCould not find JaxNav output JSONL.")
    else:
        print(f"\nAnalyzing JaxNav File: {jaxnav_files[0]}...")
        j_stats = analyze_jsonl(jaxnav_files[0])
        if j_stats:
            print("\n=== JaxNav Statistics ===")
            print(f"Total Processed Pairs: {j_stats['count']}")
            print(f"Mean Input Tokens:     {j_stats['mean_input']:.1f}")
            print(f"Mean Output Tokens:    {j_stats['mean_output']:.1f} (Includes text + thinking)")
            print(f"Mean Thinking Tokens:  {j_stats['mean_thinking']:.1f}")
            print(f"Max Thinking Tokens:   {j_stats['max_thinking']}")
            
            if x_stats and j_stats:
                print("\n=== COMPARISON ===")
                diff = x_stats['mean_thinking'] / max(1, j_stats['mean_thinking'])
                print(f"XLand required {diff:.1f}x more thinking tokens on average than JaxNav!")

if __name__ == "__main__":
    main()

"""
Prompt Evolution Visualization - Language Stage + Visual Stage

Data Source: convergence_log_*.jsonl
Features:
  1. Display Top-3 Prompts evolution across iterations
  2. Distinguish between Language and Visual stages (with different background colors)
  3. Highlight text differences between generations
  4. Generate score curves
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from difflib import Differ
from html import escape

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = "/u/home/lj0/Code/visual-feedback-meta-prompt/1-2_tpe"
MODELS = ["qwen2.5-vl-3b", "qwen2.5-vl-7b"]
DATASETS = ["miccai_nova_new", "miccai_btd_new"]
OUTPUT_DIR = "./prompt_evolution_vis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color configuration
COLOR_LANGUAGE_BG = "#E1F5FE"        # Light blue background (Language stage)
COLOR_LANGUAGE_BADGE = "#81D4FA"    # Blue badge
COLOR_VISUAL_BG = "#FFF3E0"          # Light orange background (Visual stage)
COLOR_VISUAL_BADGE = "#FFCC80"      # Orange badge
COLOR_DIVIDER = "#B0BEC5"            # Divider line

COLOR_ADDED = "#ccffcc"              # Added word (green)
COLOR_REMOVED = "#ffcccc"            # Removed word (red)

# ============================================================================
# Data Loading
# ============================================================================

def load_convergence_log(model_path):
    """Load convergence history from JSONL file"""
    log_files = sorted(Path(model_path).glob("convergence_log_*.jsonl"))
    
    if not log_files:
        print(f"⚠️ No convergence_log found in {model_path}")
        return []
    
    latest_file = log_files[-1]
    history = []
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    history.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    
    return history


def detect_stage(reason_text):
    """Detect stage from reason field"""
    reason_lower = reason_text.lower()
    
    # Check if switching from Language to Visual
    if "→" in reason_text and "visual" in reason_lower:
        return "VISUAL"
    elif "language" in reason_lower or "language_momemtum" in reason_lower:
        return "LANGUAGE"
    else:
        return "LANGUAGE"  # Default to Language stage


def extract_evolution(history):
    """
    Extract prompt evolution from convergence history
    
    Strategy: Track newly generated prompts each iteration, compare with previous best
    
    Returns:
        {
            'iterations': [0, 1, 2, ...],
            'stages': ['LANGUAGE', 'LANGUAGE', 'VISUAL', ...],
            'scores': [0.14, 0.14, 0.15, ...],
            'stds': [0.008, 0.008, ...],
            'prompts': [
                {
                    'iteration': 0,
                    'name': 'language_gen_0',
                    'text': '...',
                    'score': 0.14,
                    'is_new_generated': False,
                    'compare_with_text': None,
                    'stage': 'LANGUAGE'
                },
                ...
            ]
        }
    """
    iterations = []
    stages = []
    scores = []
    stds = []
    prompts = []
    
    last_stage = "LANGUAGE"
    best_prompt_per_iter = {}  # Track best prompt per iteration
    
    for record in history:
        iteration = record.get('iteration', 0)
        iterations.append(iteration)
        
        # Detect current stage
        reason = record.get('reason', '')
        current_stage = detect_stage(reason)
        
        # Update stage if detected
        if current_stage == "VISUAL":
            last_stage = "VISUAL"
        
        stages.append(last_stage)
        
        # Extract score info
        top_3_scores = record.get('top_3_scores', [0, 0, 0])
        avg_score = np.mean(top_3_scores)
        scores.append(avg_score)
        stds.append(record.get('top_3_std', 0))
        
        # Extract all prompts (Top-3 and Bottom-3)
        if 'prompts_snapshot' in record:
            snapshot = record['prompts_snapshot']
            top_3 = snapshot.get('top_3', [])
            bottom_3 = snapshot.get('bottom_3', [])
            
            # Record best prompt of this iteration (Rank 1)
            if top_3:
                best_prompt_per_iter[iteration] = top_3[0]
            
            # Process Top-3
            for idx, prompt_info in enumerate(top_3):
                source = prompt_info.get('source', 'unknown')
                is_new = source == 'language_momemtum_generated' or source == 'visual_momemtum'

                # Normalize prompt name based on source and iteration
                original_name = prompt_info.get('name', f'prompt_{idx}')
                if source == 'language_momemtum_generated' or 'language_momemtum_gen' in original_name:
                    display_name = f'language_momemtum_gen_{iteration}'
                elif source == 'visual_momemtum' or 'visual_momemtum_gen' in original_name:
                    display_name = f'visual_momemtum_gen_{iteration}'
                else:
                    display_name = original_name

                # Find previous iteration's prompt for comparison
                compare_with_text = None
                if is_new and iteration > 0:
                    prev_iter = iteration - 1
                    if prev_iter in best_prompt_per_iter:
                        compare_with_text = best_prompt_per_iter[prev_iter]['text']

                prompts.append({
                    'iteration': iteration,
                    'rank': idx + 1,
                    'name': display_name,
                    'text': prompt_info.get('text', ''),
                    'score': prompt_info.get('score', 0),
                    'source': source,
                    'is_new_generated': is_new,
                    'compare_with_text': compare_with_text,
                    'stage': last_stage
                })

            # Process Bottom-3 (only show newly generated candidates)
            for idx, prompt_info in enumerate(bottom_3):
                source = prompt_info.get('source', 'unknown')
                is_new = source == 'language_momemtum_generated' or source == 'visual_momemtum'

                if is_new:  # Only show newly generated
                    original_name = prompt_info.get('name', f'prompt_{idx}')
                    if source == 'language_momemtum_generated' or 'language_momemtum_gen' in original_name:
                        display_name = f'language_momemtum_gen_{iteration}_{idx+1}'
                    elif source == 'visual_momemtum' or 'visual_momemtum_gen' in original_name:
                        display_name = f'visual_momemtum_gen_{iteration}_{idx+1}'
                    else:
                        display_name = original_name
                    
                    compare_with_text = None
                    if iteration > 0:
                        prev_iter = iteration - 1
                        if prev_iter in best_prompt_per_iter:
                            compare_with_text = best_prompt_per_iter[prev_iter]['text']
                    
                    prompts.append({
                        'iteration': iteration,
                        'rank': 3 + idx + 1,  # Rank 4, 5, 6
                        'name': display_name,
                        'text': prompt_info.get('text', ''),
                        'score': prompt_info.get('score', 0),
                        'source': source,
                        'is_new_generated': is_new,
                        'compare_with_text': compare_with_text,
                        'stage': last_stage
                    })
    
    return {
        'iterations': iterations,
        'stages': stages,
        'scores': scores,
        'stds': stds,
        'prompts': prompts
    }


# ============================================================================
# Text Difference Highlighting
# ============================================================================


def highlight_text_diff(text_before, text_after):
    """
    Compare two texts and return HTML with difference highlights
    
    Args:
        text_before: Previous generation prompt text (can be None)
        text_after: Current generation prompt text
    
    Returns:
        HTML string with difference highlights
    """
    if not text_before:
        # First generation, return as is
        return escape(text_after)
    
    differ = Differ()
    diff = list(differ.compare(text_before.split(), text_after.split()))
    
    html_parts = []
    for line in diff:
        if line.startswith('- '):
            # Removed word
            word = escape(line[2:])
            html_parts.append(
                f'<span style="background-color: {COLOR_REMOVED}; text-decoration: line-through;">{word}</span>'
            )
        elif line.startswith('+ '):
            # Added word
            word = escape(line[2:])
            html_parts.append(
                f'<span style="background-color: {COLOR_ADDED}; font-weight: bold;">{word}</span>'
            )
        elif line.startswith('? '):
            # Ignore
            continue
        else:
            # Same word
            word = escape(line[2:])
            html_parts.append(word)
    
    return ' '.join(html_parts)


# ============================================================================
# HTML Table Generation
# ============================================================================


def create_prompt_evolution_html(model_name, dataset_name, evolution_data):
    """
    Generate HTML table for a single model, showing comparison of newly generated prompts with previous best
    """
    prompts = evolution_data['prompts']
    stages = evolution_data['stages']
    iterations = evolution_data['iterations']
    scores = evolution_data['scores']
    stds = evolution_data['stds']
    
    # Group by iteration
    prompts_by_iter = {}
    for p in prompts:
        iter_idx = p['iteration']
        if iter_idx not in prompts_by_iter:
            prompts_by_iter[iter_idx] = []
        prompts_by_iter[iter_idx].append(p)
    
    # Sort each group by score descending
    for iter_idx in prompts_by_iter:
        prompts_by_iter[iter_idx].sort(key=lambda x: x['score'], reverse=True)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prompt Evolution Visualization - {model_name} ({dataset_name})</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .info-bar {{
                background-color: #ecf0f1;
                padding: 12px 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 13px;
            }}
            .legend {{
                display: flex;
                gap: 25px;
                margin-bottom: 25px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
            }}
            .legend-color {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
            }}
            .iteration-header {{
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                color: white;
                padding: 15px;
                margin-top: 20px;
                margin-bottom: 0;
                border-radius: 6px 6px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .iteration-num {{
                font-size: 16px;
                font-weight: bold;
            }}
            .stage-badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            .stage-language {{
                background-color: {COLOR_LANGUAGE_BADGE};
                color: #01579b;
            }}
            .stage-visual {{
                background-color: {COLOR_VISUAL_BADGE};
                color: #e65100;
            }}
            .prompts-container {{
                border: 1px solid #ddd;
                border-top: none;
                border-radius: 0 0 6px 6px;
                margin-bottom: 15px;
            }}
            .prompt-item {{
                padding: 15px;
                border-bottom: 1px solid #eee;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .prompt-item:last-child {{
                border-bottom: none;
            }}
            .prompt-item.new-gen {{
                background-color: #fffacd;
            }}
            .prompt-item.base {{
                background-color: #f0f0f0;
            }}
            .prompt-column {{
                display: flex;
                flex-direction: column;
            }}
            .prompt-label {{
                font-weight: bold;
                font-size: 12px;
                color: #555;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .rank-badge {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: bold;
            }}
            .score-badge {{
                display: inline-block;
                background-color: #27ae60;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }}
            .prompt-name {{
                font-size: 10px;
                color: #666;
                font-family: 'Courier New', monospace;
                margin-left: auto;
            }}
            .prompt-text {{
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                line-height: 1.5;
                white-space: pre-wrap;
                word-break: break-word;
            }}
            .meta-info {{
                font-size: 10px;
                color: #999;
                margin-top: 6px;
            }}
            .arrow {{
                text-align: center;
                font-size: 20px;
                color: #3498db;
                padding: 10px 0;
            }}
            .comparison-header {{
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                padding: 8px 0;
            }}
            .divider {{
                background-color: {COLOR_DIVIDER};
                height: 2px;
                margin: 15px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Prompt Iteration Evolution Details</h1>
            
            <div class="info-bar">
                <strong>Model:</strong> {model_name} | 
                <strong>Dataset:</strong> {dataset_name} | 
                <strong>Total Iterations:</strong> {len(iterations)} |
                <strong>Methodology:</strong> Language Stage (TPE) → Compare newly generated prompts with previous best → Visual Stage (Visual Feedback)
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {COLOR_LANGUAGE_BG};"></div>
                    <span>Language Stage</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {COLOR_VISUAL_BG};"></div>
                    <span>Visual Stage</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fffacd;"></div>
                    <span>Newly Generated Prompt</span>
                </div>
                <div class="legend-item">
                    <span style="background-color: {COLOR_ADDED}; padding: 2px 6px; border-radius: 3px; font-size: 10px;">Green = New Words</span>
                </div>
                <div class="legend-item">
                    <span style="background-color: {COLOR_REMOVED}; padding: 2px 6px; border-radius: 3px; font-size: 10px; text-decoration: line-through;">Red = Removed Words</span>
                </div>
            </div>
            
            <div class="divider"></div>
    """
    
    # Display by iteration order
    for iter_idx in sorted(prompts_by_iter.keys()):
        iter_prompts = prompts_by_iter[iter_idx]
        stage = stages[iter_idx] if iter_idx < len(stages) else 'UNKNOWN'
        score = scores[iter_idx] if iter_idx < len(scores) else 0
        std = stds[iter_idx] if iter_idx < len(stds) else 0
        
        stage_class = 'language' if stage == 'LANGUAGE' else 'visual'
        stage_badge = f'<span class="stage-badge stage-{stage_class}">{stage}</span>'
        
        html += f"""
            <div class="iteration-header">
                <div class="iteration-num">🔄 Iteration #{iter_idx}</div>
                <div style="display: flex; gap: 15px; align-items: center;">
                    {stage_badge}
                    <span style="font-size: 12px;">Top-3 Score: <strong>{score:.4f}</strong> ± {std:.6f}</span>
                </div>
            </div>
            <div class="prompts-container">
        """
        
        # Separate best prompt from other candidates
        best_prompt = iter_prompts[0] if iter_prompts else None
        other_prompts = iter_prompts[1:] if len(iter_prompts) > 1 else []
        
        # 1. Display best prompt and its comparison
        if best_prompt:
            is_new = best_prompt['is_new_generated']
            compare_text = best_prompt.get('compare_with_text', None)
            
            item_class = 'new-gen' if is_new else 'base'
            rank_badge = f'<span class="rank-badge">Rank {best_prompt["rank"]}</span>'
            score_badge = f'<span class="score-badge">{best_prompt["score"]:.4f}</span>'
            
            html += f"""
                <div class="prompt-item {item_class}">
                    <div class="prompt-column">
                        <div class="prompt-label">
                            {rank_badge} {score_badge}
                            <span class="prompt-name">{best_prompt['name']}</span>
                        </div>
                        <div class="prompt-text">{escape(best_prompt['text'])}</div>
                        <div class="meta-info">Source: {best_prompt['source']}</div>
                    </div>
            """
            
            if compare_text and compare_text != best_prompt['text']:
                # Highlight differences
                highlighted = highlight_text_diff(compare_text, best_prompt['text'])
                html += f"""
                    <div class="prompt-column">
                        <div class="comparison-header">📝 vs Previous Best (Differences Highlighted)</div>
                        <div class="prompt-text">{highlighted}</div>
                        <div class="meta-info">Comparison: with Iteration #{iter_idx-1}</div>
                    </div>
                """
            else:
                html += """
                    <div class="prompt-column">
                        <div class="comparison-header">📝 (Unchanged or First Generation)</div>
                        <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; color: #999; font-size: 11px;">
                            This prompt is unchanged from the previous iteration or is the first generation
                        </div>
                    </div>
                """
            
            html += """
                </div>
            """
        
        # 2. Display other candidate prompts (if newly generated)
        new_gen_others = [p for p in other_prompts if p['is_new_generated']]
        
        if new_gen_others:
            html += f"""
                <div style="padding: 10px 15px; background-color: #fff9e6; border-bottom: 1px solid #eee; font-size: 11px; font-weight: bold; color: #b8860b;">
                    ⭐ Other Candidates Generated This Iteration ({len(new_gen_others)} total)
                </div>
            """
            
            for p in new_gen_others:
                rank_badge = f'<span class="rank-badge">Rank {p["rank"]}</span>'
                score_badge = f'<span class="score-badge">{p["score"]:.4f}</span>'
                compare_text = p.get('compare_with_text', None)
                
                html += f"""
                    <div class="prompt-item new-gen">
                        <div class="prompt-column">
                            <div class="prompt-label">
                                {rank_badge} {score_badge}
                                <span class="prompt-name">{p['name']}</span>
                            </div>
                            <div class="prompt-text">{escape(p['text'])}</div>
                        </div>
                """
                
                if compare_text:
                    highlighted = highlight_text_diff(compare_text, p['text'])
                    html += f"""
                        <div class="prompt-column">
                            <div class="comparison-header">📝 vs Previous Best</div>
                            <div class="prompt-text">{highlighted}</div>
                        </div>
                    """
                else:
                    html += """
                        <div class="prompt-column">
                            <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; color: #999; font-size: 11px;">
                                No comparison information available
                            </div>
                        </div>
                    """
                
                html += """
                    </div>
                """
        
        html += """
            </div>
        """
    
    html += """
            <div class="divider" style="margin-top: 30px;"></div>
        </div>
    </body>
    </html>
    """
    
    return html


# ============================================================================
# Score Curve Plot

def create_score_curve(model_name, dataset_name, evolution_data):
    """
    Draw score change curve with different colors for Language and Visual stages
    """
    iterations = evolution_data['iterations']
    stages = evolution_data['stages']
    scores = evolution_data['scores']
    stds = evolution_data['stds']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Prompt Optimization Curve - {model_name} ({dataset_name})', 
                 fontsize=14, fontweight='bold')
    
    # Draw background areas (distinguish stages)
    current_stage = None
    start_idx = 0
    
    for i, stage in enumerate(stages):
        if stage != current_stage and current_stage is not None:
            # Fill background for previous stage
            bg_color = COLOR_LANGUAGE_BG if current_stage == 'LANGUAGE' else COLOR_VISUAL_BG
            ax1.axvspan(start_idx - 0.5, i - 0.5, alpha=0.3, color=bg_color, zorder=0)
            ax2.axvspan(start_idx - 0.5, i - 0.5, alpha=0.3, color=bg_color, zorder=0)
            
            # Draw boundary line
            ax1.axvline(x=i - 0.5, color=COLOR_DIVIDER, linestyle='--', linewidth=2, alpha=0.5)
            ax2.axvline(x=i - 0.5, color=COLOR_DIVIDER, linestyle='--', linewidth=2, alpha=0.5)
            
            start_idx = i
        
        current_stage = stage
    
    # Last stage
    if current_stage:
        bg_color = COLOR_LANGUAGE_BG if current_stage == 'LANGUAGE' else COLOR_VISUAL_BG
        ax1.axvspan(start_idx - 0.5, iterations[-1] + 0.5, alpha=0.3, color=bg_color, zorder=0)
        ax2.axvspan(start_idx - 0.5, iterations[-1] + 0.5, alpha=0.3, color=bg_color, zorder=0)
    
    # Upper plot: Score curve + Stability band
    stds_array = np.array(stds)
    scores_array = np.array(scores)
    
    ax1.fill_between(iterations, 
                      scores_array - stds_array,
                      scores_array + stds_array,
                      alpha=0.2, color='#3498db', label='Stability Band (±Std)')
    
    ax1.plot(iterations, scores_array, 'o-', color='#3498db', 
            linewidth=2.5, markersize=6, label='Top-3 Average Score', zorder=5)
    
    ax1.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
    ax1.set_title('Score Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Lower plot: Growth rate
    velocities = np.zeros_like(scores_array)
    for i in range(1, len(scores_array)):
        if scores_array[i-1] != 0:
            velocities[i] = (scores_array[i] - scores_array[i-1]) / scores_array[i-1] * 100
    
    colors = [COLOR_LANGUAGE_BADGE if s == 'LANGUAGE' else COLOR_VISUAL_BADGE 
              for s in stages]
    
    ax2.bar(iterations, velocities, color=colors, alpha=0.7, edgecolor='none')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Δ (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_title('Growth Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add legend
    lang_patch = mpatches.Patch(facecolor=COLOR_LANGUAGE_BG, alpha=0.5, label='Language Stage')
    vis_patch = mpatches.Patch(facecolor=COLOR_VISUAL_BG, alpha=0.5, label='Visual Stage')
    ax2.legend(handles=[lang_patch, vis_patch], loc='upper right')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Main Function
# ============================================================================


def visualize_all():
    """Generate visualizations for all model + dataset combinations"""
    
    print("🚀 Starting Prompt Evolution Visualization Generation...\n")
    
    for dataset in DATASETS:
        dataset_dir = os.path.join(BASE_DIR, dataset)
        
        if not os.path.exists(dataset_dir):
            print(f"⚠️ Dataset directory not found: {dataset_dir}")
            continue
        
        print(f"📦 Processing dataset: {dataset}")
        
        for model in MODELS:
            model_path = os.path.join(dataset_dir, model)
            
            if not os.path.exists(model_path):
                print(f"  ⚠️ Model directory not found: {model_path}")
                continue
            
            print(f"  📝 Processing model: {model}")
            
            # Load data
            history = load_convergence_log(model_path)
            if not history:
                print(f"    ⚠️ No data")
                continue
            
            # Extract evolution data
            evolution_data = extract_evolution(history)
            
            # Generate HTML
            html = create_prompt_evolution_html(model, dataset, evolution_data)
            html_path = os.path.join(OUTPUT_DIR, f"prompt_evolution_{dataset}_{model}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"    ✓ HTML saved: {html_path}")
            
            # Draw score curve
            fig = create_score_curve(model, dataset, evolution_data)
            png_path = os.path.join(OUTPUT_DIR, f"score_curve_{dataset}_{model}.png")
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"    ✓ Curve saved: {png_path}")
            
            pdf_path = os.path.join(OUTPUT_DIR, f"score_curve_{dataset}_{model}.pdf")
            fig.savefig(pdf_path, bbox_inches='tight')
            print(f"    ✓ PDF saved: {pdf_path}")
            
            plt.close(fig)
    
    print(f"\n✨ Done! All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    visualize_all()

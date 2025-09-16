import json
import matplotlib.pyplot as plt
import numpy as np

def read_gc_time_data(json_file_path, value, test_filter):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    test_names = []
    for config in data.keys():
        if config in ['hybrid', 'purecap']:
            test_names.extend(data[config].keys())
    test_names = list(set(test_names))  
    
    hybrid = []
    purecap = []
    if test_filter == []:
        test_filter = test_names
    final_names = []
    for test in test_names:
        if test in data.get('hybrid', {}) and test in data.get('purecap', {}) and test in test_filter:
            hybrid.append(data['hybrid'][test].get(value, 0))
            purecap.append(data['purecap'][test].get(value, 0))
            final_names.append(test)
    return final_names, hybrid, purecap

def safe_divide(numerator, denominator):
    if denominator == 0:
        return numerator 
    else:
        return numerator / denominator

def create_normalized_gc_time_chart(json_file_path_disable, json_file_path_enable, entry='gc-time', test_filter=[], normalize=True):
    test_names, hybrid_disabled, purecap_disabled = read_gc_time_data(json_file_path_disable, entry, test_filter)
    test_names_second, hybrid_enabled, purecap_enabled = read_gc_time_data(json_file_path_enable, entry, test_filter)
    
    if test_names != test_names_second or test_names == []:
        print("Invalid test sets")
        return

    if normalize:
        hybrid_disabled_vals = [safe_divide(value, value) for value in hybrid_disabled]  
        hybrid_enabled_vals = [safe_divide(hybrid_enabled[i], hybrid_disabled[i]) for i in range(len(test_names))]
        purecap_disabled_vals = [safe_divide(purecap_disabled[i], hybrid_disabled[i]) for i in range(len(test_names))]
        purecap_enabled_vals = [safe_divide(purecap_enabled[i], hybrid_disabled[i]) for i in range(len(test_names))]
        ylabel = f'Normalized {entry.replace("-", " ").title()} (Relative to Hybrid Disabled)'
        title = f'{entry.replace("-", " ").title()} Comparison: Normalized by Hybrid Disabled Configuration'
    else:
        hybrid_disabled_vals = hybrid_disabled
        hybrid_enabled_vals = hybrid_enabled
        purecap_disabled_vals = purecap_disabled
        purecap_enabled_vals = purecap_enabled
        ylabel = f'{entry.replace("-", " ").title()} (Raw Values)'
        title = f'{entry.replace("-", " ").title()} Comparison (Raw Values)'

    fig, ax = plt.subplots(figsize=(16, 9))
    
    bar_width = 0.18
    x_pos = np.arange(len(test_names))
    
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
    
    bars1 = ax.bar(x_pos - 1.5*bar_width, hybrid_disabled_vals, bar_width, 
                  label='Hybrid - Inc Disabled', color=colors[0], 
                  edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos - 0.5*bar_width, hybrid_enabled_vals, bar_width, 
                  label='Hybrid - Inc Enabled', color=colors[1], 
                  edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x_pos + 0.5*bar_width, purecap_disabled_vals, bar_width, 
                  label='Purecap - Inc Disabled', color=colors[2], 
                  edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x_pos + 1.5*bar_width, purecap_enabled_vals, bar_width, 
                  label='Purecap - Inc Enabled', color=colors[3], 
                  edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Test Types', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_names, rotation=30, ha='right', fontsize=10, wrap=True)
    ax.legend(fontsize=12, loc='upper right')
    
    if normalize:
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    def add_value_labels(bars):
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.grid(axis='y', alpha=0.2)
    plt.ylim(0, max(max(hybrid_enabled_vals), max(purecap_disabled_vals), max(purecap_enabled_vals)) * 1.2)
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 vis.py <disable_json_file> <enable_json_file> [gc_time_entry] [normalize] [test_filter...]")
        print("Example: python3 vis.py disable.json enable.json gc-time true barnes binary_tree cfrac")
        sys.exit(1)
    
    json_file_path_disable = sys.argv[1]
    json_file_path_enable = sys.argv[2]
    
    gc_time_entry = 'gc-time'
    if len(sys.argv) > 3:
        gc_time_entry = sys.argv[3]
    
    normalize = True
    if len(sys.argv) > 4:
        if sys.argv[4].lower() in ['false', 'no', '0']:
            normalize = False
    
    test_filter = []
    if len(sys.argv) > 5:
        test_filter = sys.argv[5:]
    
    print(f"Creating chart with:")
    print(f"  Disable file: {json_file_path_disable}")
    print(f"  Enable file: {json_file_path_enable}")
    print(f"  GC time entry: {gc_time_entry}")
    print(f"  Normalize: {normalize}")
    print(f"  Test filter: {test_filter}")
    
    create_normalized_gc_time_chart(json_file_path_disable, json_file_path_enable, gc_time_entry, test_filter, normalize)

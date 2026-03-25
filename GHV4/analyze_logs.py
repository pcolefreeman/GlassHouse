import re

def analyze_log(filepath):
    amps = []
    snrs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        # Find Amplitude
        amp_match = re.search(r'amp=([0-9.]+)', lines[i])
        if amp_match:
            amps.append(float(amp_match.group(1)))
            
        # Find SNR/Contrast
        snr_match = re.search(r'snr_eig=([0-9.]+)', lines[i])
        if snr_match:
            snrs.append(float(snr_match.group(1)))

    print(f"--- Analysis for {filepath} ---")
    print(f"Max Amp: {max(amps):.3f} | Avg Amp: {sum(amps)/len(amps):.3f}")
    print(f"Max SNR: {max(snrs):.3f} | Avg SNR: {sum(snrs)/len(snrs):.3f}\n")

# Run the analysis
analyze_log("EMPTYROOMBASELINE_run_sar.txt")
analyze_log("person_center_sar.txt")
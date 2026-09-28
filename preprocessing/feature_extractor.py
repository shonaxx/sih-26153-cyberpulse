# Calculates entropy, IAT, and flag ratios
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Any

class WindowFeatureExtractor:
    """
    Calculates the statistical and structural features for a single time window.
    Converts raw packet lists into the numerical State Vector X[t].
    """
    def __init__(self, window_duration_sec: float = 10.0):
        self.window_duration = max(window_duration_sec, 0.001) # Prevent div by zero

    def _calculate_entropy(self, items: List[Any]) -> float:
        """Calculates normalized Shannon Entropy (0.0 to 1.0)."""
        if not items:
            return 0.0
        
        counts = Counter(items)
        total = len(items)
        entropy = 0.0
        
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
            
        # Normalize by maximum possible entropy for the number of unique items
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def extract_features(self, packets: List[Dict[str, Any]], active_flows_count: int) -> Dict[str, float]:
        """
        Processes a list of packet dictionaries and returns the state feature vector.
        """
        num_packets = len(packets)
        if num_packets == 0:
            # Return baseline/zero state if network is completely silent
            return {"total_packets": 0, "total_bytes": 0, "syn_rate": 0.0} # Truncated for brevity if empty

        total_bytes = 0
        syn_count, fin_count, rst_count, udp_count, icmp_count = 0, 0, 0, 0, 0
        dst_ports = []
        src_ips = []
        packet_lengths = []

        for pkt in packets:
            total_bytes += pkt['wire_len']
            packet_lengths.append(pkt['wire_len'])
            dst_ports.append(pkt['dst_port'])
            src_ips.append(pkt['src_ip'])

            # Count protocols
            if pkt['proto'] == 17: udp_count += 1
            if pkt['proto'] == 1: icmp_count += 1

            # Count TCP flags (if it's a TCP packet)
            flags = pkt.get('tcp_flags', "")
            if 'S' in flags: syn_count += 1
            if 'F' in flags: fin_count += 1
            if 'R' in flags: rst_count += 1

        # Calculate Rates (per second)
        syn_rate = syn_count / self.window_duration
        rst_rate = rst_count / self.window_duration
        
        # Calculate Ratios
        syn_ratio = syn_count / num_packets
        udp_ratio = udp_count / num_packets
        
        # Calculate Entropies (crucial for detecting scans and sweeps)
        dst_port_entropy = self._calculate_entropy(dst_ports)
        
        # Calculate Micro-statistics
        mean_packet_len = float(np.mean(packet_lengths))
        std_packet_len = float(np.std(packet_lengths)) if num_packets > 1 else 0.0

        # Construct the final feature vector dictionary
        features = {
            "total_packets": float(num_packets),
            "total_bytes": float(total_bytes),
            "active_flows": float(active_flows_count),
            "syn_rate": round(syn_rate, 4),
            "rst_rate": round(rst_rate, 4),
            "syn_ratio": round(syn_ratio, 4),
            "udp_ratio": round(udp_ratio, 4),
            "unique_dst_ports": float(len(set(dst_ports))),
            "unique_src_ips": float(len(set(src_ips))),
            "dst_port_entropy": round(dst_port_entropy, 4),
            "mean_packet_length": round(mean_packet_len, 2),
            "std_packet_length": round(std_packet_len, 2)
        }
        
        return features

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing Feature Extractor ---")
    
    # Simulate an attacker running an Nmap Port Scan (Lots of SYN packets to different ports)
    simulated_scan_packets = []
    for port in range(1000, 1050): # Scanning 50 different ports
        simulated_scan_packets.append({
            "wire_len": 60, "src_ip": "10.0.0.99", "dst_ip": "192.168.1.10",
            "src_port": 55555, "dst_port": port, "proto": 6, "tcp_flags": "S"
        })
        
    extractor = WindowFeatureExtractor(window_duration_sec=10.0)
    
    # We pass the packets, plus a dummy 'active flows' count from the tracker
    extracted_state = extractor.extract_features(simulated_scan_packets, active_flows_count=50)
    
    print("\nExtracted Feature Vector X[t] for Simulated Port Scan:")
    for feature, value in extracted_state.items():
        # Highlighting the behavioral signatures of an attack
        if feature == "dst_port_entropy" and value > 0.8:
            print(f"[*] {feature}: {value}  <-- HIGH ENTROPY DETECTED (Port Scan Indicator)")
        elif feature == "syn_ratio" and value > 0.5:
            print(f"[*] {feature}: {value}  <-- HIGH SYN RATIO DETECTED (Sweep Indicator)")
        else:
            print(f"    {feature}: {value}")
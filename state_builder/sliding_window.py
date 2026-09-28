# Aggregates features over sliding windows
import os
import sys
import csv
from typing import List, Dict, Any

# Add the root project directory to Python's path so it can find our other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom pipeline components
from preprocessing.pcap_parser import stream_pcap
from preprocessing.flow_tracker import FlowTracker
from preprocessing.feature_extractor import WindowFeatureExtractor

class SlidingWindowBuilder:
    """
    Orchestrates the ingestion of packets, groups them into overlapping 
    time windows, and exports the feature state vectors X[t] to CSV.
    """
    def __init__(self, window_size_sec: float = 10.0, step_size_sec: float = 5.0):
        self.window_size = window_size_sec
        self.step_size = step_size_sec
        
        self.flow_tracker = FlowTracker(timeout_sec=120.0)
        self.feature_extractor = WindowFeatureExtractor(window_duration_sec=self.window_size)
        
        self.packet_buffer: List[Dict[str, Any]] = []
        self.current_window_start = None

    def process_pcap_to_csv(self, pcap_path: str, output_csv_path: str):
        """Streams a PCAP, builds sliding windows, and saves X[t] vectors to disk."""
        print(f"[*] Processing PCAP: {pcap_path}")
        print(f"[*] Window Size: {self.window_size}s | Step Size: {self.step_size}s")
        
        extracted_windows = []
        is_first_packet = True
        
        for pkt in stream_pcap(pcap_path):
            if is_first_packet:
                # Align our very first window to the timestamp of the first packet
                self.current_window_start = pkt['timestamp']
                is_first_packet = False
                
            # 1. Update the flow tracker
            self.flow_tracker.process_packet(pkt)
            
            # 2. Check if this packet pushes us past the current window boundary
            while pkt['timestamp'] >= self.current_window_start + self.window_size:
                # Snapshot the current state!
                active_flows = self.flow_tracker.get_active_flows_count()
                features = self.feature_extractor.extract_features(self.packet_buffer, active_flows)
                
                # Tag it with chronological metadata
                features['window_start'] = round(self.current_window_start, 4)
                features['window_end'] = round(self.current_window_start + self.window_size, 4)
                extracted_windows.append(features)
                
                # Slide the window forward by step_size (5 seconds)
                self.current_window_start += self.step_size
                
                # Flush inactive flows to save RAM
                self.flow_tracker.flush_expired_flows(self.current_window_start)
                
                # Discard packets from the buffer that are now outside the new window
                self.packet_buffer = [
                    p for p in self.packet_buffer 
                    if p['timestamp'] >= self.current_window_start
                ]
                
            # 3. Add the current packet to our rolling buffer
            self.packet_buffer.append(pkt)

        # ---------------------------------------------------------
        # FLUSH FIX: Process the final leftover packets when the file ends
        # ---------------------------------------------------------
        if self.packet_buffer:
            active_flows = self.flow_tracker.get_active_flows_count()
            features = self.feature_extractor.extract_features(self.packet_buffer, active_flows)
            
            features['window_start'] = round(self.current_window_start, 4)
            features['window_end'] = round(self.packet_buffer[-1]['timestamp'], 4)
            extracted_windows.append(features)
        # ---------------------------------------------------------

        # Write all extracted windows to a CSV file
        if extracted_windows:
            self._write_to_csv(extracted_windows, output_csv_path)
        else:
            print("[!] No windows extracted. PCAP might be too short.")

    def _write_to_csv(self, windows: List[Dict[str, Any]], output_path: str):
        """Safely writes dictionaries to a CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        headers = list(windows[0].keys())
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for window in windows:
                writer.writerow(window)
        
        print(f"[+] Successfully wrote {len(windows)} state vectors to {output_path}")


# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    # Point this to the dummy PCAP we generated in Phase 1
    test_pcap_in = "data/raw_pcaps/test_traffic.pcap"
    test_csv_out = "data/processed_csv/test_features.csv"
    
    if not os.path.exists(test_pcap_in):
        print(f"[!] Please run preprocessing/pcap_parser.py first to generate {test_pcap_in}")
    else:
        print("--- Testing Sliding Window State Builder ---")
        builder = SlidingWindowBuilder(window_size_sec=10.0, step_size_sec=5.0)
        builder.process_pcap_to_csv(test_pcap_in, test_csv_out)
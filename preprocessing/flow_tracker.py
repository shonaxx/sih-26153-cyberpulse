# In-memory 5-tuple bidirectional flow tracking
from typing import Dict, Any, List
import time

class FlowTracker:
    """
    Maintains active network flows in memory using a 5-tuple hash.
    Aggregates packet-level stats into flow-level metrics.
    """
    def __init__(self, timeout_sec: float = 120.0):
        self.active_flows = {}
        self.timeout_sec = timeout_sec

    def _generate_flow_key(self, pkt: Dict[str, Any]) -> str:
        """Generates a bidirectional-safe string key for the 5-tuple."""
        ips = sorted([pkt['src_ip'], pkt['dst_ip']])
        ports = sorted([str(pkt['src_port']), str(pkt['dst_port'])])
        return f"{pkt['proto']}-{ips[0]}:{ports[0]}-{ips[1]}:{ports[1]}"

    def process_packet(self, pkt: Dict[str, Any]) -> None:
        """Ingests a packet and updates the corresponding flow statistics."""
        # Skip packets lacking transport layer ports (e.g., pure IP fragments)
        if pkt['src_port'] == 0 and pkt['dst_port'] == 0:
            return

        flow_key = self._generate_flow_key(pkt)
        current_time = pkt['timestamp']

        if flow_key not in self.active_flows:
            # Initialize a new flow
            self.active_flows[flow_key] = {
                "flow_key": flow_key,
                "start_time": current_time,
                "last_seen": current_time,
                "packet_count": 1,
                "total_bytes": pkt['wire_len'],
                "is_active": True
            }
        else:
            # Update existing flow
            flow = self.active_flows[flow_key]
            flow["packet_count"] += 1
            flow["total_bytes"] += pkt['wire_len']
            flow["last_seen"] = current_time

    def get_active_flows_count(self) -> int:
        return len(self.active_flows)

    def flush_expired_flows(self, current_time: float) -> int:
        """
        Removes flows that haven't seen a packet within the timeout window.
        Crucial for preventing RAM exhaustion during long PCAP reads.
        """
        expired_keys = [
            key for key, flow in self.active_flows.items()
            if (current_time - flow['last_seen']) > self.timeout_sec
        ]
        for key in expired_keys:
            del self.active_flows[key]
            
        return len(expired_keys)

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing Flow Tracker ---")
    tracker = FlowTracker(timeout_sec=2.0)
    
    # Simulate packets arriving over time
    dummy_packets = [
        {"src_ip": "10.0.0.1", "dst_ip": "8.8.8.8", "src_port": 5000, "dst_port": 443, "proto": 6, "wire_len": 100, "timestamp": 1.0},
        {"src_ip": "8.8.8.8", "dst_ip": "10.0.0.1", "src_port": 443, "dst_port": 5000, "proto": 6, "wire_len": 250, "timestamp": 1.5},
        {"src_ip": "192.168.1.5", "dst_ip": "1.1.1.1", "src_port": 3333, "dst_port": 53, "proto": 17, "wire_len": 60, "timestamp": 1.8},
    ]

    for p in dummy_packets:
        tracker.process_packet(p)
    
    print(f"Active flows after ingestion: {tracker.get_active_flows_count()} (Expected: 2)")
    print("Flow Database:", tracker.active_flows)
    
    # Simulate time passing to trigger garbage collection
    print("\nSimulating 3 seconds passing...")
    expired = tracker.flush_expired_flows(current_time=5.0)
    print(f"Expired flows cleared from RAM: {expired} (Expected: 2)")
    print(f"Active flows remaining: {tracker.get_active_flows_count()} (Expected: 0)")
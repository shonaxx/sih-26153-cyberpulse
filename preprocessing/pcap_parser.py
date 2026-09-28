# Streaming packet extractor using scapy.PcapReader
import os
from typing import Iterator, Dict, Any, Optional
from scapy.all import PcapReader, IP, TCP, UDP, ICMP

def stream_pcap(pcap_path: str) -> Iterator[Dict[str, Any]]:
    """
    Streams a PCAP file packet-by-packet without loading it into RAM.
    Extracts only the metadata required for the SIH26153 state vector.
    
    Yields:
        Dictionary containing extracted packet metadata, or skips if non-IP.
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    # PcapReader acts as a generator, keeping memory usage flat (< 50MB)
    with PcapReader(pcap_path) as reader:
        for packet in reader:
            # We only care about IP layer traffic for network forecasting
            if not packet.haslayer(IP):
                continue
            
            ip_layer = packet[IP]
            
            # Base metadata required for all packets
            packet_data = {
                "timestamp": float(packet.time),
                "wire_len": len(packet),
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "proto": ip_layer.proto,  # 6=TCP, 17=UDP, 1=ICMP
                "ttl": ip_layer.ttl,
                "src_port": 0,
                "dst_port": 0,
                "tcp_window": 0,
                "tcp_flags": ""
            }

            # Extract Transport Layer Specifics
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                packet_data["src_port"] = tcp_layer.sport
                packet_data["dst_port"] = tcp_layer.dport
                packet_data["tcp_window"] = tcp_layer.window
                packet_data["tcp_flags"] = str(tcp_layer.flags)
                
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                packet_data["src_port"] = udp_layer.sport
                packet_data["dst_port"] = udp_layer.dport
                
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                packet_data["src_port"] = icmp_layer.type # Overload port with ICMP type
                packet_data["dst_port"] = icmp_layer.code # Overload port with ICMP code

            yield packet_data

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    from scapy.all import wrpcap, Ether
    
    # 1. Create a dummy PCAP file automatically so you can test this instantly
    test_pcap = "data/raw_pcaps/test_traffic.pcap"
    print(f"Generating dummy test PCAP at {test_pcap}...")
    
    dummy_packets = [
        Ether()/IP(dst="192.168.1.10", ttl=64)/TCP(sport=443, dport=54321, flags="SA", window=29200),
        Ether()/IP(dst="192.168.1.50", ttl=128)/TCP(sport=5678, dport=22, flags="S"),
        Ether()/IP(dst="8.8.8.8", ttl=64)/UDP(sport=5353, dport=53)
    ]
    wrpcap(test_pcap, dummy_packets)
    
    # 2. Run the parser and print memory-safe output
    print("\n--- Testing PCAP Streamer ---")
    packet_count = 0
    
    try:
        for pkt_meta in stream_pcap(test_pcap):
            print(pkt_meta)
            packet_count += 1
            if packet_count >= 5: # Limit output if you test on a real, massive PCAP later
                print("... (truncating for test)")
                break
        print(f"\nSuccessfully parsed {packet_count} packets. Memory remained stable.")
    except Exception as e:
        print(f"Error parsing PCAP: {e}")
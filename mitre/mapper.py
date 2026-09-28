# Maps telemetry features and risk scores to MITRE Tactics
import os
import sys
from typing import Dict, Any, List

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MitreMapper:
    """
    Translates model logits and raw telemetry into actionable MITRE ATT&CK tactics.
    Includes a Defense-in-Depth Rule Engine to override model hallucinations.
    """
    def __init__(self):
        # SIH26153 Standard Mapping
        self.stage_map = {
            0: {"tactic": "Normal Traffic", "id": "TA0000", "techniques": []},
            1: {"tactic": "Reconnaissance", "id": "TA0043", "techniques": ["T1046 (Network Service Scanning)", "T1595 (Active Scanning)"]},
            2: {"tactic": "Initial Access", "id": "TA0001", "techniques": ["T1110 (Brute Force)", "T1190 (Exploit Public-Facing App)"]},
            3: {"tactic": "Lateral Movement", "id": "TA0008", "techniques": ["T1021 (Remote Services)", "T1570 (Lateral Tool Transfer)"]},
            4: {"tactic": "Command & Control", "id": "TA0011", "techniques": ["T1071 (Application Layer Protocol)", "T1573 (Encrypted Channel)"]},
            5: {"tactic": "Exfiltration", "id": "TA0010", "techniques": ["T1048 (Exfil Over Alt Protocol)"]}
        }

    def evaluate_trajectory(self, predicted_class_idx: int, risk_score: float, current_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Takes the raw ML predictions and validates them against network invariants.
        Returns a dictionary ready for the SOC Dashboard.
        """
        # 1. Base Mapping
        mapped_stage = self.stage_map.get(predicted_class_idx, self.stage_map[0])
        confidence_modifier = "High"
        override_applied = False
        
        # 2. Validation Rule Engine (Sanity Checks)
        
        # Rule A: Ignore low-risk predictions even if the stage classifier guessed an attack
        if risk_score < 0.40 and predicted_class_idx != 0:
            mapped_stage = self.stage_map[0]
            override_applied = True
            confidence_modifier = "Low (Downgraded due to low risk score)"
            
        # Rule B: Lateral Movement requires internal-to-internal traffic
        elif predicted_class_idx == 3:
            # Assuming 'internal_to_internal_ratio' exists, but we fallback safely if not
            internal_ratio = current_features.get("internal_to_internal_ratio", 0.0)
            if internal_ratio < 0.1:
                mapped_stage = self.stage_map[2] # Downgrade to Initial Access
                override_applied = True
                confidence_modifier = "Low (Overridden: No internal traffic detected)"

        # Rule C: Reconnaissance validation via Port Entropy
        elif predicted_class_idx == 1:
            entropy = current_features.get("dst_port_entropy", 0.0)
            if entropy > 0.7:
                confidence_modifier = "High (Confirmed by severe port entropy)"

        # 3. Format the final output
        return {
            "mapped_tactic": mapped_stage["tactic"],
            "tactic_id": mapped_stage["id"],
            "suggested_techniques": mapped_stage["techniques"],
            "escalation_probability": round(risk_score, 4),
            "confidence_modifier": confidence_modifier,
            "rule_override_applied": override_applied
        }


# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing MITRE ATT&CK Mapping Engine ---")
    mapper = MitreMapper()
    
    # Scenario 1: The ML model predicts an attack correctly (Reconnaissance)
    print("\n[Scenario 1: High Entropy Reconnaissance]")
    scenario_1_features = {"dst_port_entropy": 0.85, "syn_rate": 45.0}
    result_1 = mapper.evaluate_trajectory(predicted_class_idx=1, risk_score=0.88, current_features=scenario_1_features)
    for k, v in result_1.items(): print(f"  {k}: {v}")
        
    # Scenario 2: The ML model hallucinates Lateral Movement on External Traffic
    print("\n[Scenario 2: ML Hallucination (Lateral Movement with 0% Internal Traffic)]")
    scenario_2_features = {"internal_to_internal_ratio": 0.0, "total_bytes": 1000}
    result_2 = mapper.evaluate_trajectory(predicted_class_idx=3, risk_score=0.92, current_features=scenario_2_features)
    for k, v in result_2.items(): print(f"  {k}: {v}")
    
    # Scenario 3: Low risk threshold prevents false alarms
    print("\n[Scenario 3: Model slightly confused, low risk threshold]")
    scenario_3_features = {"dst_port_entropy": 0.1}
    result_3 = mapper.evaluate_trajectory(predicted_class_idx=4, risk_score=0.25, current_features=scenario_3_features)
    for k, v in result_3.items(): print(f"  {k}: {v}")
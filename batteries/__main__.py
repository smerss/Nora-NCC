"""CLI runner for Nora evaluation batteries: python -m batteries [name] [options]"""
import os
import runpy
import sys

BATTERY_MAP = {
    "agi": "batteries.agi_battery",
    "agi_battery": "batteries.agi_battery",
    "architecture": "batteries.architecture_battery",
    "architecture_battery": "batteries.architecture_battery",
    "battery": "batteries.battery",
    "counterfactual": "batteries.battery",
    "bridge": "batteries.bridge_battery",
    "bridge_battery": "batteries.bridge_battery",
    "causal": "batteries.causal_battery",
    "causal_battery": "batteries.causal_battery",
    "consciousness": "batteries.consciousness_battery",
    "consciousness_battery": "batteries.consciousness_battery",
    "crossmodal": "batteries.crossmodal_battery",
    "crossmodal_battery": "batteries.crossmodal_battery",
    "evolution": "batteries.evolution_battery",
    "evolution_battery": "batteries.evolution_battery",
    "external": "batteries.external_battery",
    "external_battery": "batteries.external_battery",
    "heldout": "batteries.heldout_battery",
    "heldout_battery": "batteries.heldout_battery",
    "heldout_negatif": "batteries.heldout_negatif",
    "heldout_sealed": "batteries.heldout_sealed",
    "heldout_sealed2": "batteries.heldout_sealed2",
    "heldout_sealed3": "batteries.heldout_sealed3",
    "integrate": "batteries.integrate_battery",
    "integrate_battery": "batteries.integrate_battery",
    "law": "batteries.law_battery",
    "law_battery": "batteries.law_battery",
    "metacog": "batteries.metacog_battery",
    "metacog_battery": "batteries.metacog_battery",
    "object": "batteries.object_battery",
    "object_battery": "batteries.object_battery",
    "persistence": "batteries.persistence_battery",
    "persistence_battery": "batteries.persistence_battery",
    "planning": "batteries.planning_battery",
    "planning_battery": "batteries.planning_battery",
    "practical_language": "batteries.practical_language_battery",
    "practical_language_battery": "batteries.practical_language_battery",
    "survival": "batteries.survival_battery",
    "survival_battery": "batteries.survival_battery",
    "translation": "batteries.translation_battery",
    "translation_battery": "batteries.translation_battery",
    "workspace": "batteries.workspace_battery",
    "workspace_battery": "batteries.workspace_battery",
    "selftest": "batteries.selftest",
    "solo": "batteries.solo_gorev",
    "solo_gorev": "batteries.solo_gorev",
    "sozce": "batteries.sozce_siniflandirma_degerlendirme",
}


def print_help():
    print("Nora Battery Runner")
    print("Usage: python -m batteries <battery_name> [args...]")
    print("\nAvailable batteries:")
    for k in sorted(set(BATTERY_MAP.keys())):
        if not k.endswith("_battery") and not k.startswith("heldout_"):
            print(f"  - {k:22} ({BATTERY_MAP[k]})")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help", "list"):
        print_help()
        sys.exit(0)
    name = sys.argv[1]
    mod_name = BATTERY_MAP.get(name)
    if not mod_name:
        print(f"Unknown battery: {name}")
        print_help()
        sys.exit(1)

    # Pin hash seed before sub-execution so hash seed execv does not drop arguments
    os.environ["_NORA_HASH_PINNED"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"

    sys.argv = [mod_name] + sys.argv[2:]
    runpy.run_module(mod_name, run_name="__main__")


if __name__ == "__main__":
    main()

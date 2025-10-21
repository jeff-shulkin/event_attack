from event_attack import EventAttack
import pathlib
import argparse


def main(args):
    attack_config_path = args.config

    print("Initializing attack...")
    attack = EventAttack(attack_config_path=attack_config_path)
    print("Finished setting up attack.")

    # Start flickering with a given image
    print("Starting image flicker...")
    attack.flicker()
    print("Ending image flicker...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack test argument parser")
    parser.add_argument("-c", "--config", type=pathlib.Path)
    args = parser.parse_args()
    main(args)
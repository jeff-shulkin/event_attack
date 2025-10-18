from event_attack import EventAttack
import argparse


def main(args):
    injected_image_filename = args.injected_image_filename
    carrier_image_filename = args.carrier_image_filename
    flicker_fps = args.fps

    print("Initializing attack...")
    attack = EventAttack(
        inject_img_path=injected_image_filename,
        carrier_img_path = carrier_image_filename,
        attack_method="linear",
        fps=flicker_fps
    )
    print("Finished setting up attack.")

    # Start flickering with a given image
    print("Starting image flicker...")
    attack.flicker(None, None)
    print("Ending image flicker...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack test argument parser")
    parser.add_argument("-i", "--injected_image_filename", type=str, required=True)
    parser.add_argument("-c", "--carrier_image_filename", type=str, required=True)
    parser.add_argument("-f", "--fps", type=int, required=True)
    args = parser.parse_args()
    main(args)
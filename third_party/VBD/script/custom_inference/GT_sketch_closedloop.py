"""Closed-loop or open-loop VBD inference with GT future sketch conditioning."""

import utils_closed_loop as utils


def _sketch(batch):
    # Use the GT future in the batch as conditioning, no manual sketching.
    return utils.snapshot_conditioning(batch)


def main() -> None:
    args = utils.parse_args()

    # Closed-loop uses receding horizon; open-loop falls back to one-shot prediction.
    if args.closedloop:
        print("\n🔄 Running CLOSED-LOOP simulation with GT sketch conditioning")
        utils.run_closedloop_simulation(args, _sketch, sketch_name='GT Sketch')
    else:
        print("\n➡️  Running OPEN-LOOP simulation with GT sketch conditioning")
        utils.run_simulation(args, _sketch, sketch_name='GT Sketch')


if __name__ == '__main__':
    main()

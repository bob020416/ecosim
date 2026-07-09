import argparse
from typing import Tuple

import utils_closed_loop as utils


def _parse_target_agents(value: str) -> Tuple[int, ...]:
    if not value:
        return ()
    agents = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        agents.append(int(part))
    return tuple(agents)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VBD custom sketch inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ctrl_ckpt_path", type=str, required=True)
    parser.add_argument("--test_mode", type=str, default="diffusion", choices=["diffusion", "prior"])
    parser.add_argument("--save_simulation", action="store_true")
    parser.add_argument("--plot_control_samples", action="store_true")
    parser.add_argument("--use_training_format", action="store_true", default=True)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    parser.add_argument("--only_controlnet", action="store_true")
    parser.add_argument("--output_dir", type=str, default="vbd_custom_sketch")
    parser.add_argument("--closedloop", action="store_true")
    parser.add_argument("--horizon_steps", type=int, default=11)
    parser.add_argument("--total_steps", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_candidates", type=int, default=4)

    parser.add_argument(
        "--sketch",
        choices=["forward", "right_turn", "static"],
        default="right_turn",
        help="Synthetic sketch template to apply to target agents.",
    )
    parser.add_argument("--target_agents", type=str, default="0,1")
    parser.add_argument("--forward_distance", type=float, default=5.0)
    parser.add_argument("--turn_distance", type=float, default=5.0)
    parser.add_argument("--turn_angle_deg", type=float, default=60.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    target_agents = _parse_target_agents(args.target_agents)
    if not target_agents:
        raise RuntimeError("--target_agents must include at least one valid agent index.")

    def _sketch(batch):
        if args.sketch == "right_turn":
            return utils.apply_right_turn_sketch(
                batch,
                forward_distance=args.forward_distance,
                turn_distance=args.turn_distance,
                turn_angle_deg=args.turn_angle_deg,
                target_agents=target_agents,
            )
        distance = 0.0 if args.sketch == "static" else args.forward_distance
        return utils.apply_forward_sketch(
            batch,
            distance=distance,
            target_agents=target_agents,
        )

    label = {
        "forward": f"Forward {args.forward_distance:g}m",
        "right_turn": f"Right Turn {args.turn_angle_deg:g}deg",
        "static": "Static",
    }[args.sketch]

    if args.closedloop:
        print(f"\nRunning CLOSED-LOOP simulation with custom VBD sketch: {label}")
        utils.run_closedloop_simulation(args, _sketch, sketch_name=label)
    else:
        print(f"\nRunning OPEN-LOOP simulation with custom VBD sketch: {label}")
        utils.run_simulation(args, _sketch, sketch_name=label)


if __name__ == "__main__":
    main()

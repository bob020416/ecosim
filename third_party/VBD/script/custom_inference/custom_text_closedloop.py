import argparse
from typing import List, Tuple

import torch

import utils_closed_loop as utils_cl
import utils as utils_ol


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


def _parse_sentences(text: str, delimiter: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if delimiter and delimiter in text:
        return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [text]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VBD custom text inference")
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
    parser.add_argument("--output_dir", type=str, default="vbd_custom_text")
    parser.add_argument("--closedloop", action="store_true")
    parser.add_argument("--horizon_steps", type=int, default=11)
    parser.add_argument("--total_steps", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_candidates", type=int, default=4)

    parser.add_argument("--custom_text", type=str, required=True)
    parser.add_argument("--custom_text_delim", type=str, default="||")
    parser.add_argument(
        "--target_agents",
        type=str,
        default="0",
        help="Comma-separated agent indices to receive text conditioning.",
    )
    parser.add_argument("--print_text_samples", type=int, default=3)
    return parser


def _no_sketch(_batch):
    return None, []


class CustomTextAttacher:
    def __init__(
        self,
        sentences: List[str],
        target_agents: Tuple[int, ...],
        print_samples: int,
    ) -> None:
        if not sentences:
            raise RuntimeError("--custom_text cannot be empty.")
        self.sentences = sentences
        self.target_agents = target_agents
        self.print_samples = max(0, int(print_samples))
        self.samples_printed = 0

    def attach(self, batch: dict, scenario_data: dict) -> None:
        agents_history = batch.get("agents_history")
        if agents_history is None:
            raise RuntimeError("Batch missing agents_history; cannot attach text prompts.")
        num_agents = int(agents_history.shape[1])
        target_agents = [idx for idx in self.target_agents if 0 <= idx < num_agents]
        if not target_agents:
            raise RuntimeError("No valid --target_agents found for this scenario.")

        prompts = ["" for _ in range(num_agents)]
        mask = torch.zeros(num_agents, dtype=torch.bool)
        for order, agent_idx in enumerate(target_agents):
            sentence = self.sentences[min(order, len(self.sentences) - 1)]
            sentence = sentence.replace("<ego>", f"agent {agent_idx}")
            sentence = sentence.replace("<agent>", f"agent {agent_idx}")
            prompts[agent_idx] = sentence
            mask[agent_idx] = True

        batch["text_prompt"] = prompts
        batch["text_prompt_mask"] = mask

        if self.samples_printed < self.print_samples:
            self.samples_printed += 1
            scenario_id = scenario_data.get("scenario_id", batch.get("scenario_id", "unknown"))
            print(f"[custom_text_closedloop] scenario_id={scenario_id}")
            for agent_idx in target_agents[:3]:
                print(f"[custom_text_closedloop]   agent {agent_idx}: {prompts[agent_idx]}")


def main() -> None:
    args = _build_parser().parse_args()
    sentences = _parse_sentences(args.custom_text, args.custom_text_delim)
    target_agents = _parse_target_agents(args.target_agents)
    attacher = CustomTextAttacher(
        sentences=sentences,
        target_agents=target_agents,
        print_samples=args.print_text_samples,
    )

    if args.closedloop:
        print("\nRunning CLOSED-LOOP simulation with custom VBD text conditioning")
        utils_cl.run_closedloop_simulation(
            args,
            _no_sketch,
            sketch_name="Custom Text",
            batch_preprocess_fn=attacher.attach,
        )
    else:
        print("\nRunning OPEN-LOOP simulation with custom VBD text conditioning")
        utils_ol.run_simulation(
            args,
            _no_sketch,
            sketch_name="Custom Text",
            batch_preprocess_fn=attacher.attach,
        )


if __name__ == "__main__":
    main()

import torch
import torch.quantization
import os
import json
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.utils import load_checkpoint, save_checkpoint
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load parameters from JSON file
    with open("checkpoints/parameters.json") as f:
        params = json.load(f)

    # Initialize the MeshGraphNet model
    model = MeshGraphNet(
        params["infeat_nodes"],
        params["infeat_edges"],
        2,  # Output dimension
        processor_size=cfg.architecture.processor_size,
        hidden_dim_node_encoder=cfg.architecture.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=cfg.architecture.hidden_dim_edge_encoder,
        hidden_dim_processor=cfg.architecture.hidden_dim_processor,
        hidden_dim_node_decoder=cfg.architecture.hidden_dim_node_decoder,
    )

    model = model.to(device)
    model.eval()

    # Load the pre-trained model checkpoint
    load_checkpoint(os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name), models=model, device=device)

    # Perform Post-Training Quantization (PTQ)
    # Dynamic quantization reduces model size and increases inference speed by converting weights to int8.
    # This process is suitable for models where most operations are matrix multiplications, such as fully connected layers.
    # We specifically target Linear and Convolutional layers for quantization to balance performance and accuracy.
    model_int8 = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # Save the quantized model to a file
    quantized_model_path = os.path.join(cfg.checkpoints.ckpt_path, "model_quantized.pt")
    torch.save(model_int8.state_dict(), quantized_model_path)

if __name__ == "__main__":
    main()

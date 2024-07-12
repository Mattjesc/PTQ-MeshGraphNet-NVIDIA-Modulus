import torch
import matplotlib.pyplot as plt
import json
import os
import time
import numpy as np  # Import numpy
from generate_dataset import generate_normalized_graphs
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
from omegaconf import DictConfig
import hydra

def denormalize(tensor, mean, stdv):
    return tensor * stdv + mean

class MGNRolloutQuantized:
    def __init__(self, logger, cfg):
        """Initialization for performing the rollout phase on the geometry specified in
        'config.yaml' (testing.graph) and computing the error using a quantized model."""
        
        self.device = "cpu"  # Set device to CPU for quantized model inference
        self.logger = logger
        logger.info(f"Using {self.device} device")

        # Load parameters from JSON file
        params = json.load(open("checkpoints/parameters.json"))

        # Generate normalized graphs from raw dataset
        norm_type = {"features": "normal", "labels": "normal"}
        graphs, params = generate_normalized_graphs(
            "raw_dataset/graphs/",
            norm_type,
            cfg.training.geometries,
            params["statistics"],
        )
        graph = graphs[list(graphs)[0]]

        infeat_nodes = graph.ndata["nfeatures"].shape[1] + 1
        infeat_edges = graph.edata["efeatures"].shape[1]
        nout = 2
        nodes_features = [
            "area",
            "tangent",
            "type",
            "T",
            "dip",
            "sysp",
            "resistance1",
            "capacitance",
            "resistance2",
            "loading",
        ]

        edges_features = ["rel_position", "distance", "type"]

        params["infeat_nodes"] = infeat_nodes
        params["infeat_edges"] = infeat_edges
        params["out_size"] = nout
        params["node_features"] = nodes_features
        params["edges_features"] = edges_features
        params["rate_noise"] = 100
        params["rate_noise_features"] = 1e-5
        params["stride"] = 5

        self.graphs = graphs

        # Load the quantized model
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

        # Apply dynamic quantization to specific layers
        # This reduces model size and improves inference speed by converting weights to int8.
        self.model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d}, dtype=torch.qint8
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load the quantized model checkpoint
        self.model.load_state_dict(torch.load("checkpoints/model_quantized.pt", map_location=self.device))

        self.params = params
        self.var_identifier = {"p": 0, "q": 1}

    def compute_average_branches(self, graph, flowrate):
        """Average flowrate over branch nodes."""
        branch_id = graph.ndata["branch_id"].cpu().detach().numpy()
        bmax = np.max(branch_id)
        for i in range(bmax + 1):
            idxs = np.where(branch_id == i)[0]
            rflowrate = torch.mean(flowrate[idxs])
            flowrate[idxs] = rflowrate

    def predict(self, graph_name):
        """Perform rollout phase for a single graph in the dataset."""
        graph = self.graphs[graph_name]
        graph = graph.to(self.device)
        self.graph = graph

        ntimes = graph.ndata["pressure"].shape[-1]
        nnodes = graph.ndata["pressure"].shape[0]

        self.pred = torch.zeros((nnodes, 2, ntimes), device=self.device)
        self.exact = graph.ndata["nfeatures"][:, 0:2, :].to(self.device)
        self.pred[:, 0:2, 0] = graph.ndata["nfeatures"][:, 0:2, 0].to(self.device)

        inmask = graph.ndata["inlet_mask"].bool().to(self.device)
        invar = graph.ndata["nfeatures"][:, :, 0].clone().squeeze().to(self.device)
        efeatures = graph.edata["efeatures"].squeeze().to(self.device)
        nf = torch.zeros((nnodes, 1), device=self.device)
        
        start = time.time()
        for i in range(ntimes - 1):
            invar[:, -1] = graph.ndata["nfeatures"][:, -1, i].to(self.device)
            nf[inmask, 0] = graph.ndata["nfeatures"][inmask, 1, i + 1].to(self.device)
            nfeatures = torch.cat((invar, nf), 1)
            pred = self.model(nfeatures, efeatures, graph).detach()
            invar[:, 0:2] += pred
            invar[inmask, 1] = graph.ndata["nfeatures"][inmask, 1, i + 1].to(self.device)
            self.compute_average_branches(graph, invar[:, 1])

            self.pred[:, :, i + 1] = invar[:, 0:2]

        end = time.time()
        self.logger.info(f"Rollout took {end - start} seconds!")

    def denormalize(self):
        """Denormalize predicted and exact pressure and flow rate values."""
        self.pred[:, 0, :] = denormalize(
            self.pred[:, 0, :],
            self.params["statistics"]["pressure"]["mean"],
            self.params["statistics"]["pressure"]["stdv"],
        )
        self.pred[:, 1, :] = denormalize(
            self.pred[:, 1, :],
            self.params["statistics"]["flowrate"]["mean"],
            self.params["statistics"]["flowrate"]["stdv"],
        )
        self.exact[:, 0, :] = denormalize(
            self.exact[:, 0, :],
            self.params["statistics"]["pressure"]["mean"],
            self.params["statistics"]["pressure"]["stdv"],
        )
        self.exact[:, 1, :] = denormalize(
            self.exact[:, 1, :],
            self.params["statistics"]["flowrate"]["mean"],
            self.params["statistics"]["flowrate"]["stdv"],
        )

    def compute_errors(self):
        """Compute errors in pressure and flow rate. This function must be called
        after 'predict'."""
        bm = torch.reshape(self.graph.ndata["branch_mask"], (-1, 1, 1)).to(self.device)
        bm = bm.repeat(1, 2, self.pred.shape[2])
        diff = (self.pred - self.exact) * bm
        errs = torch.sum(torch.sum(diff**2, axis=0), axis=1)
        norm = torch.sum(torch.sum((self.exact * bm) ** 2, axis=0), axis=1)
        errs = errs / norm
        errs = torch.sqrt(errs)

        self.logger.info(f"Relative error in pressure: {errs[0] * 100}%")
        self.logger.info(f"Relative error in flowrate: {errs[1] * 100}%")

    def plot(self, idx):
        """Creates plot of pressure and flow rate at the node specified with the
        idx parameter."""
        load = self.graph.ndata["nfeatures"][0, -1, :].to(self.device)
        p_pred_values = []
        q_pred_values = []
        p_exact_values = []
        q_exact_values = []

        bm = self.graph.ndata["branch_mask"].bool().to(self.device)

        nsol = self.pred.shape[2]
        for isol in range(nsol):
            if load[isol] == 0:
                p_pred_values.append(self.pred[bm, 0, isol][idx].cpu())
                q_pred_values.append(self.pred[bm, 1, isol][idx].cpu())
                p_exact_values.append(self.exact[bm, 0, isol][idx].cpu())
                q_exact_values.append(self.exact[bm, 1, isol][idx].cpu())

        plt.figure()
        ax = plt.axes()

        ax.plot(p_pred_values, label="pred")
        ax.plot(p_exact_values, label="exact")
        ax.legend()
        plt.xlabel('Time (s)')  # Adding X-axis label
        plt.ylabel('Pressure (mmHg)')  # Adding Y-axis label
        plt.savefig("pressure_quantized.png", bbox_inches="tight")

        plt.figure()
        ax = plt.axes()

        ax.plot(q_pred_values, label="pred")
        ax.plot(q_exact_values, label="exact")
        ax.legend()
        plt.xlabel('Time (s)')  # Adding X-axis label
        plt.ylabel('Flow Rate (mL/s)')  # Adding Y-axis label
        plt.savefig("flowrate_quantized.png", bbox_inches="tight")

# Define the main function to run the inference
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger = PythonLogger("MGNRolloutQuantized")
    rollout = MGNRolloutQuantized(logger, cfg)
    rollout.predict(cfg.testing.graph)
    rollout.denormalize()
    rollout.compute_errors()
    rollout.plot(idx=5)  # Plot results at a specific node

if __name__ == "__main__":
    main()

# Post-Training Quantization for MeshGraphNet using NVIDIA Modulus: Cardiovascular Flow Simulation Implementation

## Overview

This project explores the application of post-training quantization (PTQ) to the MeshGraphNet model for simulating cardiovascular flow. The goal is to reduce computational demands, enabling deployment on resource-limited devices without significant loss of accuracy. The study details the quantization process, evaluates performance, and compares the quantized model with the original.

## Prerequisites

1. **Clone the NVIDIA Modulus Repository**:
    Follow the instructions provided in the NVIDIA Modulus repository to set up the Modulus framework:
    - [NVIDIA Modulus Repository](https://github.com/NVIDIA/modulus)
    - Refer to the README in the Modulus repo for detailed setup instructions.

2. **Prepare the Bloodflow Example**:
    Navigate to the `examples/healthcare/bloodflow_1d_mgn` directory in the Modulus repository and prepare the example as per the instructions provided in the respective README file.

## Adding Quantized Files

After setting up the NVIDIA Modulus repository and preparing the bloodflow example, add the quantized files from this project to the `examples/healthcare/bloodflow_1d_mgn` directory:

1. **Copy Quantized Files**:
    Copy the `train_quantized.py` and `inference_quantized.py` files to the `examples/healthcare/bloodflow_1d_mgn` directory.

## Workflow

1. **Run `train.py`**: This script trains the original MeshGraphNet model.
    ```sh
    python train.py
    ```

2. **Run `inference.py`**: This script performs inference using the trained original model and generates output graphs.
    ```sh
    python inference.py
    ```

3. **Run `train_quantized.py`**: This script applies post-training quantization to the MeshGraphNet model.
    ```sh
    python train_quantized.py
    ```

4. **Run `inference_quantized.py`**: This script performs inference using the quantized model and generates output graphs.
    ```sh
    python inference_quantized.py
    ```

5. **Compare Results**: Compare the results of `pressure.png` and `flowrate.png` from the original model with `pressure_quantized.png` and `flowrate_quantized.png` from the quantized model.

## Example Results

| Metric                      | Original Model | PTQ Model   |
|-----------------------------|----------------|-------------|
| Relative error in pressure  | 0.83%          | 17.91%      |
| Relative error in flow rate | 4.00%          | 33.35%      |
| Rollout time                | 0.653 seconds  | 0.535 seconds |


Figure 1: Original Model Rollout and Relative Error Values

![image](https://github.com/user-attachments/assets/4f78dcf9-ffb0-4b23-b713-2c9e8a926962)

Figure 2: Quantized (PTQ) Model Rollout and Relative Error Values

![image](https://github.com/user-attachments/assets/2a0c97e6-7f40-40bf-a527-c9f4a41cd4b9)

Figure 3: Original Model Pressure Graph

![image](https://github.com/user-attachments/assets/bac7e8a2-5dc2-434e-b720-5a4c7fda808d)

Figure 4: Quantized (PTQ) Model Pressure Graph

![image](https://github.com/user-attachments/assets/ef9543b6-8ef0-49c3-b56d-2cdc56c37c60)

Figure 5: Original Model Flow Rate Graph

![image](https://github.com/user-attachments/assets/9472b233-2a4e-4ba9-9f60-d7067f651c55)

Figure 6: Quantized (PTQ) Model Flow Rate Graph

![image](https://github.com/user-attachments/assets/0f53f947-9c65-4905-85c2-0f7d13be6f30)


## Justifications and Explanations

### Why Quantize Using PTQ?

Post-training quantization reduces the model size and memory usage, making it feasible to run large models on resource-limited hardware. PTQ specifically optimizes models for efficient inference, with the trade-off between computational efficiency and accuracy.

### Hardware Considerations

The quantization process and model evaluations are performed on a single RTX 3090 GPU. Different hardware setups, particularly variations in CPU capabilities, may yield varying outcomes in terms of training/inference speed and performance.

## Acknowledgements

Special thanks to:
- [Authors and Contributors](https://arxiv.org/abs/2010.03409) of the MeshGraphNet model and the underlying datasets.
- [NVIDIA Modulus Team](https://developer.nvidia.com/modulus) for their framework and support.
- [Vascular Model Repository](https://vascularmodel.com/) for providing the dataset.

## References

- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
- [PD-Quant: Post-Training Quantization based on Prediction Difference Metric](https://arxiv.org/abs/2212.07048)
- [Recipes for Post-Training Quantization of Deep Neural Networks](https://arxiv.org/abs/2007.00893)

## Citation

```yaml
cff-version: 1.2.0
message: If you use this software, please cite it as below.
title: NVIDIA Modulus
version: 0.6.0
authors:
  - family-names: NVIDIA Modulus Team
url: https://github.com/NVIDIA/modulus/tree/main

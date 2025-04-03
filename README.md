# Welcome, Developer! 👋

**New to compiler development?** You're in the right place! This repository is your map to the frontier of AI compiler development. Follow along as we track how hardware and software innovations converge, paving the way for next-gen machine learning systems.

- 📅 **Chronological Insights** – Explore major breakthroughs in order  
- ⚙️ **Compiler Foundations** – Learn fundamentals from real projects  
- 🚀 **Optimization Tips** – Find hardware/software sweet spots  
- 🤝 **Community-Driven** – Share knowledge and grow with others  

---

# 🔥 AI Hardware Evolution Timeline

Dive into the machines powering modern AI—where silicon meets neural magic. Below is a chronological journey, showcasing how each architecture pushed the boundaries of efficiency, scalability, and performance.

01. **DianNao (2014) by Tianshi Chen et al.**  
   > 🏆 First dedicated neural accelerator—tiny but powerful, inspiring future designs  
   > 📝 [Paper (ASPLOS '14)](https://dl.acm.org/doi/10.1145/2654822.2541967)  
   > 🎞️ [Video](https://youtu.be/IA5lCVywS0I)  
   > 🔧 [OpenDianNao Implementation](https://github.com/accelergy-community/opendiannao)  
   > 📚 [DianNao Family Retrospective](https://people.csail.mit.edu/emer/papers/2020.06.ieeemicro.diannao.pdf)  

02. **DaDianNao (2014) by Tianshi Chen et al.**  
   > 🏆 Scaled-up DianNao for datacenters, tackling large-scale DNNs with multi-chip designs  
   > 📝 [Paper (ASPLOS '14)](https://dl.acm.org/doi/10.1145/2593069.2593077)  
   > 🔧 [Architecture Analysis Toolkit](https://github.com/hpca-accelerator/accelerator-simulator)  
   > 📚 [Multi-Chip Interconnect Study](https://ieeexplore.ieee.org/document/9139512)  

03. **ShiDianNao (2015) by Zidong Du et al.**  
   > 🏆 Integrated vision accelerator with sensor-processor fusion—ideal for edge devices  
   > 📝 [Paper (ISCA '15)](https://dl.acm.org/doi/10.1145/2749469.2750389)  
   > 🔧 [FPGA Demo Project](https://github.com/HewlettPackard/diannao-series)  
   > 📚 [Edge AI Survey](https://arxiv.org/abs/2206.08063)  

04. **EIE (Efficient Inference Engine) (2016) by Song Han et al.**  
   > 🏆 First hardware for sparse neural networks, leveraging pruning to reduce compute  
   > 📝 [Paper (FPGA '16)](https://arxiv.org/abs/1603.01670)  
   > 🎞️ [Video](https://youtu.be/vouEMwDNopQ)  
   > 🔧 [SparseNN Toolkit](https://github.com/facebookresearch/SparseNN)  
   > 📚 [Pruning Tutorial](https://nervanasystems.github.io/distiller/algo_pruning/)  

05. **Eyeriss (2016) by Yu-Hsin Chen et al.**  
   > 🏆 Energy-efficient spatial architecture for CNNs, optimizing data reuse  
   > 📝 [Paper (ISSCC '16)](https://ieeexplore.ieee.org/document/7783717)  
   > 🎞️ [Video](https://youtu.be/brhOo-_7NS4)  
   > 📚 [MIT Course Materials](https://eyeriss.mit.edu)  
   > 📚 [Data Reuse Patterns Analysis](https://arxiv.org/abs/1807.07928)  

06. **TPU (2017) by Norman Jouppi et al. (Google)**  
   > 🏆 Revolutionized ML acceleration with 8-bit ops, dominating inference workloads  
   > 📝 [Paper (ISCA '17)](https://dl.acm.org/doi/10.1145/3079856.3080246)  
   > 🎞️ [Architecture Deep Dive](https://youtu.be/F6fLwGE83Cw)  
   > 🎞️ [Performance Analysis](https://youtu.be/eyAjbgkBdjU)  
   > 📚 [Google Cloud TPU Docs](https://cloud.google.com/tpu/docs/system-architecture)  
   > 📚 [TPU Performance Guide](https://arxiv.org/abs/2007.05509)  

07. **Cerebras WSE (2019) by Sean Lie et al.**  
   > 🏆 Wafer-scale engine—56× larger than GPUs, redefining training scalability  
   > 📝 [Paper (Hot Chips '19)](https://arxiv.org/abs/2104.13857)  
   > 🎞️ [Wafer-Scale Demo](https://youtu.be/L1kCqgxRNDY)  
   > 🔧 [Cerebras SDK](https://github.com/Cerebras/modelzoo)  
   > 📚 [Scaling Laws Analysis](https://arxiv.org/abs/2203.15556)  

08. **Sparse Tensor Cores (2020) by Ashish Mishra et al. (NVIDIA)**  
   > 🏆 2× speedups on GPUs via structured sparsity—critical for massive models  
   > 📝 [Paper (ASPLOS '20)](https://dl.acm.org/doi/10.1145/3373376.3378533)  
   > 🔧 [CUDA Samples](https://github.com/NVIDIA/sparse-tensor-cores)  
   > 📚 [Sparsity Training Guide](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-in-ampere/)  

09. **Graphcore IPU (2020) by Graphcore**  
   > 🏆 Innovative data-flow approach, enabling fine-grained parallelism for AI models  
   > 📝 [IPU Whitepaper](https://www.graphcore.ai/technology)  
   > 🎞️ [Demo](https://youtu.be/kpgf-_qekT4)  
   > 🔧 [Graphcore SDK](https://github.com/graphcore)

10. **Habana Gaudi (2020) by Habana Labs (Intel)**  
   > 🏆 Scalable training architecture with native support for mixed precision  
   > 📝 [Gaudi Overview](https://habana.ai/gaudi-training-accelerator/)  
   > 🎞️ [Inference/Training Demo](https://youtu.be/BDyFQU5a_nY)  
   > 🔧 [Habana GitHub](https://github.com/HabanaAI)  

11. **SambaNova DataScale (2021) by SambaNova Systems**  
   > 🏆 Reconfigurable dataflow architecture; flexible for various model types  
   > 📝 [SambaNova Tech Brief](https://sambanova.ai/resources/)  
   > 🎞️ [DataScale Overview](https://youtu.be/3BdxEftbK50)  
   > 🔧 [SambaFlow SDK](https://github.com/sambanova)  

12. **TPU v4 (2021) by Google Research**  
   > 🏆 Optical interconnects + dynamic reconfiguration = supercomputer-scale ML  
   > 📝 [Paper (2023)](https://arxiv.org/abs/2304.01433)  
   > 📚 [Optical Networking Primer](https://research.google/pubs/pub51601/)  
   > 📚 [Dynamic ML Systems Survey](https://arxiv.org/abs/2305.03782)  

13. **Groq TSP (2022) by Groq**  
   > 🏆 Single-core SIMD model with deterministic execution—optimized for low-latency inference  
   > 📝 [Groq Technology](https://groq.com/product/architecture/)  
   > 🎞️ [Demo](https://youtu.be/DE_6mdG0h6Q)  
   > 🔧 [Groq Compiler Docs](https://groq-docs.readthedocs-hosted.com/)  

14. **IBM NorthPole (2023) by Dharmendra Modha et al.**  
   > 🏆 Brain-inspired design with 25× energy efficiency gains over GPUs  
   > 📝 [Paper (Science '23)](https://www.science.org/doi/10.1126/science.adh1174)  
   > 🔧 [Neuro-Symbolic SDK](https://github.com/IBM/northpole-sdk)  
   > 📚 [Neuromorphic Computing Review](https://www.nature.com/articles/s41586-020-2786-7)  

---

# ⚙️ Core Compiler Milestones

From symbolic graph compilers to MLIR-based ecosystems, these projects laid the foundation for modern AI compilation.

01. **Theano (2010)**  
   > 🏆 Pioneered symbolic differentiation & early GPU acceleration in Python  
   > 📝 [Original Paper](https://conference.scipy.org/proceedings/scipy2010/pdfs/bastien.pdf)  
   > 📚 [Theano Legacy Docs](https://theano-pymc.readthedocs.io/en/latest/)  
   > 🔧 [PyMC3 (Theano Spin-off)](https://github.com/pymc-devs/pymc)  
   > 📚 [TensorFlow/Theano Comparison](https://arxiv.org/abs/1605.02688)  

02. **Halide (2012)**  
   > 🏆 Separated image-processing algorithms from their optimizations  
   > 📝 [PLDI Paper](https://dl.acm.org/doi/10.1145/2491956.2462176)  
   > 🔧 [Halide GitHub](https://github.com/halide/Halide)  
   > 📚 [Adobe's Halide Use Cases](https://research.adobe.com/news/halide-in-production/)  
   > 📚 [MIT 6.815 Course Material](https://stellar.mit.edu/S/course/6/fa15/6.815/)  

03. **TensorRT (2016)**  
   > 🏆 Kernel fusion & INT8 quantization for NVIDIA GPUs  
   > 📚 [Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)  
   > 🔧 [Triton Inference Server](https://github.com/triton-inference-server/server)  
   > 📚 [NGC TensorRT Containers](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=dateModifiedDESC&query=tensorrt)  

04. **XLA (2017)**  
   > 🏆 TensorFlow’s JIT compiler: aggressive op fusion for TPU/GPU/CPU  
   > 📚 [XLA Architecture](https://www.tensorflow.org/xla)  
   > 🔧 [JAX: XLA for NumPy](https://github.com/google/jax)  
   > 📚 [XLA vs. TVM Benchmark](https://arxiv.org/abs/2105.07585)  

05. **Glow (2018)**  
   > 🏆 Ahead-of-time compilation and unified quantization flows  
   > 📝 [Glow Paper](https://arxiv.org/abs/1805.00907)  
   > 🔧 [ONNX-Glow Integration](https://github.com/onnx/onnx/blob/main/docs/Glow.md)  
   > 📚 [PyTorch Mobile Compiler](https://pytorch.org/mobile/home/)  

06. **TVM (2018)**  
   > 🏆 ML-driven kernel optimization & cross-platform deployment  
   > 📝 [OSDI Paper](https://www.usenix.org/conference/osdi18/presentation/chen)  
   > 📚 [Apache TVM Docs](https://tvm.apache.org/docs/)  
   > 🔧 [TVM Model Zoo](https://github.com/apache/tvm-model-zoo)  
   > 📚 [OctoML Production Stack](https://octoml.ai/technology/)  

07. **MLIR (2021)**  
   > 🏆 Compiler infrastructure revolution: multi-level IR for heterogeneous computing  
   > 📝 [MLIR Whitepaper](https://arxiv.org/abs/2002.11054)  
   > 📚 [LLVM MLIR Tutorial](https://mlir.llvm.org/docs/Tutorials/)  
   > 🔧 [Torch-MLIR Project](https://github.com/llvm/torch-mlir)  
   > 🎞️ [Google I/O MLIR Talk](https://youtu.be/qzljG6DKgic)  

08. **Triton (2021)**  
   > 🏆 Python-like abstraction for writing efficient GPU kernels  
   > 📝 [PLDI Paper](https://dl.acm.org/doi/abs/10.1145/3485486)  
   > 🔧 [Triton GitHub](https://github.com/openai/triton)  
   > 📚 [Triton vs. CUDA Benchmarks](https://openreview.net/forum?id=9XSIgzaFOT)  

09. **IREE (2021)**  
   > 🏆 End-to-end MLIR-based compiler/runtime for mobile & edge  
   > 📚 [IREE Architecture](https://google.github.io/iree/)  
   > 📱 [Android Deployment Guide](https://github.com/openxla/iree/blob/main/docs/developers/android.md)  
   > 🔧 [Vulkan Backend Demo](https://github.com/iree-org/iree-samples)  

10. **TorchDynamo (2022)**  
   > 🏆 PyTorch graph capture in dynamic Python → FX Graph  
   > 📝 [TorchDynamo Paper](https://arxiv.org/abs/2205.08543)  
   > 🔧 [TorchInductor](https://pytorch.org/docs/stable/dynamo/)  
   > 📚 [Hugging Face Integration](https://huggingface.co/docs/diffusers/optimization/torchdynamo)  

11. **Mojo (2023) by Modular**  
   > 🏆 High-performance superset of Python designed for AI/HPC with low-level control  
   > 📝 [Mojo Documentation](https://docs.modular.com/mojo)  
   > 🎞️ [Mojo Introduction Talk](https://youtu.be/HJfvDVVqbNs)  
   > 🔧 [Mojo Playground](https://github.com/modularml/mojo)  

---

## ⚡ Pro Tips

- **Hands-On is Gold**: Jump into MLIR, TVM, or Triton. Break stuff and learn by doing!
- **Watch Hardware+Software**: Each new GPU/TPU/accelerator release unlocks fresh compiler challenges.
- **Embrace Sparsity**: Pruned or sparse-weight models can see huge speedups on specialized hardware.
- **Stay Curious**: Tech evolves fast—subscribe to relevant conferences (ISCA, MICRO, PLDI, etc.) and follow open-source repos.

---
_**Happy hacking and optimizing!**_

# Vast.ai recommendation for full ModernBERT-large training

For the pre-norm transformer, complete encoder fine-tuning, 1,024-token limit and full RLCD loss across the collection **excluding MS MARCO and CommitPackFT**, my default recommendation is **one NVIDIA RTX 6000 Ada, 48 GB, on-demand**. This is a capacity recommendation, not a measured GPU throughput ranking. The model has about 421 million trainable parameters; additional VRAM provides room to test larger question batches and reduce recomputation.

## Current public offers

The read-only Vast.ai API search returned these verified single-GPU offers with reliability at least 99%, at least 64 GB allocated host RAM, eight effective CPU cores, 100 GB available disk capacity and 500 Mbps reported download speed. See the timestamp and exact query in [the offer snapshot](vast_training_offers.json). Availability and prices change.

| GPU | Advertised VRAM | Lowest sampled API hourly rate | Use |
| --- | ---: | ---: | --- |
| RTX 4090 | 24 GB | $0.363 | Lower-cost pilot; increase batch size only after measuring memory |
| RTX 5090 | 32 GB | $0.450 | Candidate if the selected CUDA/PyTorch/attention stack is validated on Blackwell |
| RTX 6000 Ada | 48 GB | $0.724 | Default full-run choice with additional batching room |
| L40S | 48 GB | $0.842 | Alternative 48 GB Ada card |
| A100 PCIe | 40 GB | $0.508 | Attractive alternative; the cheapest sampled hosts advertise CUDA ceiling 12.2, so check driver compatibility with this project's recent dependencies |
| A100 SXM4 | 80 GB | $0.975 | More room for larger batches; measure whether throughput justifies the price |

These are API `dph_total` values at the search/default storage allocation. The disk-capacity filter does **not** allocate 100 GB. Recheck total pricing with the chosen storage allocation and bandwidth charges before renting. At the sampled RTX 6000 Ada rate, 24 hours is approximately $17.38 at the quoted allocation, before additional storage/transfer charges.

The sampled RTX 6000 Ada offer ID was **54444926**, with 64,393 MiB allocated host RAM, 16 effective CPU cores, approximately 816 Mbps download, advertised CUDA ceiling 13.2 and about nine days remaining maximum contract duration. Its reliability score was 0.9900369; prefer a higher score if an otherwise comparable listing appears. The API reports about 46,068 MiB usable GPU memory versus the 48 GB advertised card capacity. This is a snapshot, not a reservation.

## Instance and training setup

- Choose one full GPU, on-demand, a verified host, preferably at least 99.5% reliability, 64 GB host RAM, 8–16 effective CPU cores and 100 GB allocated SSD/NVMe storage. Select a maximum rental duration longer than the measured job estimate.
- Use a CUDA PyTorch image whose runtime is supported by the host driver and this repository's dependency versions. The repo requires PyTorch >=2.14 and Transformers >=5.17; an arbitrary old image is not sufficient.
- Keep all encoder parameters trainable, context 1,024, the pre-norm independently initialized transformer and full RLCD loss. Start a CUDA smoke with BF16 autocast and FP32 parameters/optimizer, checkpointing, batch size four and question microbatch size four; these are starting points to measure, not validated fit guarantees.
- Preserve effective batch size when initially comparing different microbatch settings. A larger effective batch is a training change and needs validation, not only a memory check.
- Benchmark larger question microbatches, reduced checkpointing and CUDA attention implementations only after the baseline passes. The current encoder explicitly uses SDPA; FlashAttention-2 is an optional code/config change and has not been enabled or tested in this repo.
- Keep bounded streaming and isolated validation/final evaluation. Current full-encoder presets still use bounded training pools; full-corpus streaming needs a separate configuration.

Our M1-specific cache clearing and one-question recomputation settings prioritize memory. Keeping all of those settings unchanged on a larger CUDA GPU can waste capacity. This recommendation does not claim that buying a faster GPU alone realizes its full throughput.

## Runtime and cost

Run a 10–20 minute representative training-only CUDA benchmark across short, long and multi-question sources. Include tokenization, streaming, backward passes, optimizer updates, validation and checkpoint overhead when estimating the final job. Do not infer an NVIDIA runtime from the M1 speed or advertised TFLOPS alone.

No instance was created or billed by this recommendation. Search used public endpoints without credentials.

Sources: [Vast.ai offer API](https://docs.vast.ai/api-reference/search/search-offers), [Vast.ai instance selection and price breakdown](https://docs.vast.ai/guides/instances/choosing/find-and-rent), [NVIDIA RTX 6000 Ada specifications](https://www.nvidia.com/en-us/products/workstations/rtx-6000/), [ModernBERT attention documentation](https://huggingface.co/docs/transformers/model_doc/modernbert). Used the local [Vast.ai skill](/Users/matheoledevehat/.codex/skills/rawveg-vastai-api/SKILL.md) to discover the official search workflow.

[![](https://img.shields.io/badge/buy_me_a_coffee-gold?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/pedro.cardoso)
[![]()]()
# web-physics-tuner
Python tool optimizing soft-body physics (ropes, banners) for web games. 
Uses AI tuning to generate lightweight JSON configs, ensuring 60fps performance 
without runtime ML overhead. Simulates mass-spring systems with Verlet integration. 
Auto-tunes stiffness/damping to match target motion. Exports production-ready config files.

```mermaid
---
title: Pipeline
---
flowchart TD
Data[Image or Video] --> Denoiser

subgraph ImageProcessing [Image Processing]
    direction LR
    Denoiser(Denoiser) --> Skeletonizer(Skeletonizer) --> NodeGen(Node Generator)
end

NodeGen --> Nodes["Nodes' positions and velocities\nper frame in json format"]
Nodes --> Optimizer

subgraph ReverseEngine [Reverse Engine]
    Optimizer(Optimizer)
end
```


## Physics Engine & Rendering
- [x] Basic particle simulation (spring system, fixed/free particles)
- [x] UI for particles and connections
- [x] Constraints: gravity, dampening, rigid connections
- [x] Helpers for constraints and mesh presets (e.g., flags)
- [x] Torsion force implementation

## Dataset Generation
- [ ] Synthetic data pipeline:
    - [ ] Randomize base string properties
    - [ ] Randomize initial positions and velocities
- [x] JSON export system
- [ ] Simulation parameters: gravity, elastic/dampening forces, mass
- [ ] Data loaders for train/test pipelines
- [ ] Normalize all the required data

## Optics & Object Tracking
- [x] Blur filter for noise reduction
- [x] Video point extraction pipeline
- [ ] Demo video verification
- [ ] Implement point correlation/tracking across frames (minimizing overall Euclidean displacement)

## Inverse Problem Optimization
- [ ] ML pipeline for parameter retrieval
- [ ] Model to predict next positions based on current state and $\theta$ parameters
- [ ] Train on synthetic data

- [x] Set the normalization setup
- [ ] Generate final dataset
- [ ] Focus on the NN architecture

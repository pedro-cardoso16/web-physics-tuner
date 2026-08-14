## Physics engine and game rendering
- [x] Test with **two** particles only (spring system)
    - [x] One particle fixed
    - [x] One particle free to move (1 dimension)
- [x] UI
    - [x] Display dots
    - [x] Display line segments connecting the two particles

- [x] Add more constraints to the particles
    - [x] gravity effect
    - [x] dampening effect
    - [x] rigid connection (distance is fixed, any movement along the axis must displace both (force is divided between particles))

- [x] Create a helper for rapidly setting references and constraints, it should 
    be easy to apply gravity, dampening, spring and rigid connection between particles

- [x] Create helper, for creating ready-made presets, like flag, of 2d mesh.

---

## Optics and object tracking
- [x] Make the blurring to avoid grainy noise from detection.
- [ ] Make the video point extraction pipeline (only information is position per frame).
- [ ] Create demo video and verify visually if the process is coherent.
 
## Inverse problem optimization
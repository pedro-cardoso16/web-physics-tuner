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

- [x] Add torsion force *(not perfect, but usable)*
---

## Dataset generation

- [ ] Prepare the synthetic data generation pipeline
  - [ ] Use the base string preset and randomize some properties

- [ ] Save the information to a json file
- [ ]  

## Optics and object tracking
- [x] Make the blurring to avoid grainy noise from detection.
- [x] Make the video point extraction pipeline (only information is position per frame).
- [ ] Create demo video and verify visually if the process is coherent.
 
> In order to create the connection I think the best method is to see the 
> minimum global displacement.
> Basically, run for each frame idependently.
> From the first frame see the closest connection to the next one. That is to calculate the minimum overall
> displacement between iterations

I have a 2 sets of coordinates one before and other after. How can I calculate the
correlation that will have the minimum overall displacement

$$
P_0 = \{(x_{0,0},y_{0,0}),(x_{0,1},y_{0,2}),\dots\}\\
P_1 = \{(x_{1,0},y_{1,0}),(x_{1,1},y_{1,2}),\dots\}
$$
We have the euclidian distance $D(i,j) = ||z_{1,j} - z_{0,i}|| = \sqrt{(x_{1,j} - x_{0,i})^2 + (y_{1,j} - y_{0,i})^2}$

We want to minimize the sum of the combinations of our euclidian distances:

$$
\min_{i,j}\left[ \sum_{i,j \ \in \{0,1,2,...,n-1\} \ } D(i,j) \right]
$$
We want a permutation that willl give the best overall result, but we only need to permutate one set the $P_1$, so.
$$
{\min}_{I}\left[ \sum_{I} D(I) \right] = {\min}_I \sum_{k = 0}^{n-1} D\left(z_{0,k}, z_{1,I(k)}\right)^2
$$

$$
I = (0,1,2,\dots,n-1)
$$
Is there a way to turn this into a constant stuff? The problem with discrete 
test is that we will have $n!$ combinations with that





## Inverse problem optimization
- [ ] Create the machine learning pipeline for the parameters retrieval.
- [ ] Train the model to identify the next position based on the information. Iderectly the other node information
    helps to calculate correctly.

The idea here is to train on synthetic data that learns how to predict the next 
position based on the input and theta parameters.



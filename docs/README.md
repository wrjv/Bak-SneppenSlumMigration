## A 2D Bak-Sneppen Slum Migration Model

Slum migration is a phenomenon mostly studied by social sciences and a topic difficult to quantify and make tangible. By modifying a 2D [Bak-Sneppen model](https://en.wikipedia.org/wiki/Bak%E2%80%93Sneppen_model), often used to model evolution, we attempt to generate the migration patterns. We show that by using only a few rules in a simple cellular automaton complex migration patterns start to occur. 

<img src="https://familyincluded.com/wp-content/uploads/2016/07/17252250461_11094da4cd_k.jpg" width="100%" alt="An indian slum."/>

## The Basic Model

Bak-Sneppen does not have empty cells, but this modified model does.

Each time step, the following steps are taken:
1. Select the cell with the lowest fitness within the model.
2. Select a random empty cell and give this cell a random new fitness.
3. Empty the previously selected cell at 1) and lower the cells within Neumann Neighbourhood by a certain factor.

How did we do the fitness?

## The Destination

### The Optimal Slum

### The Optimal Location within a Slum

## New Slum Locations

<video width="320" height="240" controls>
  <source src="http://slum.life/videos/slum_barebones.mp4" type="video/mp4">
</video>

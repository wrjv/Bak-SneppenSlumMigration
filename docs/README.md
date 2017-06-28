<style>
 img {
 	margin: 0 auto;
 	display: block;
 }
</style>

## A 2D Bak-Sneppen Slum Migration Model

Slum migration is a phenomenon mostly studied by social sciences and a topic difficult to quantify and make tangible. By modifying a 2D [Bak-Sneppen model](https://en.wikipedia.org/wiki/Bak%E2%80%93Sneppen_model), often used to model evolution, we attempt to generate the migration patterns. We show that by using only a few rules in a simple cellular automaton complex migration patterns start to occur. 

<img src="https://familyincluded.com/wp-content/uploads/2016/07/17252250461_11094da4cd_k.jpg" width="100%" alt="An indian slum."/>

## The Basic Model

Bak-Sneppen does not have empty cells, but this modified model does.

Each time step, the following steps are taken:
<ol type="A">
<li>Select the cell with the lowest fitness within the model.</li>
<li>Select a random empty cell and give this cell a new fitness using the formula mentioned below.</li>
<img src="https://latex.codecogs.com/gif.latex?new\_fitness&space;=&space;old\_fitness&space;&plus;&space;|\mathcal{N}(0,&space;\frac{1&space;-&space;old\_fitness}{3})|" style="border: 0; outline: 0; box-shadow: none;"/>
<li>Empty the previously selected cell at A. and lower the cells within the [von Neumann Neighbourhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood) by a certain factor.</li>
</ol>

<img src="http://slum.life/images/bak-sneppen_expl.png" width="100%" alt="Basic Bax-Sneppen steps."/>

How did we do the fitness?

## The Destination

### The Optimal Slum

### The Optimal Location within a Slum

## New Slum Locations

<img src="http://slum.life/videos/slum_barebones.gif" width="100%" alt="An indian slum."/>


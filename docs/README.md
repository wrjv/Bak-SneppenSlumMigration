<head>

 <script

 src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js">
 </script>
 <script>

 $(document).ready(function()
 {
     $(".gif_container").click(
         function()
         {
           var src = $(this).children(".gif").attr("src");
           if(src.includes("png")) {
            $(this).children(".gif").attr("src", src.replace(/\.png$/i, ".gif"));
            $(this).children(".overlay").hide();
            $(this).children(".play_button").hide();
           } else {
            $(this).children(".gif").attr("src", src.replace(/\.gif$/i, ".png"));
            $(this).children(".overlay").show();
            $(this).children(".play_button").show()
           }
         });

 });

 document.getElementById("project_title").innerHTML = "Complex Systems Simulation";
</script>

<link rel="stylesheet" type="text/css" href="http://slum.life/style.css">
</head>

## A 2D Bak-Sneppen Slum Migration Model

Slum migration is a phenomenon mostly studied by social sciences and a topic difficult to quantify and make tangible. By modifying a 2D [Bak-Sneppen model](https://en.wikipedia.org/wiki/Bak%E2%80%93Sneppen_model), often used to model evolution, we attempt to generate the migration patterns. We show that by using only a few rules in a simple cellular automaton complex migration patterns start to occur.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/ba/Slums_in_Caracas%2C_Venezuela.jpg" width="100%"/>

## The Basic Model

The basic version of the Slum Migration Model has only some slight modifications with respect to the basic 2D Bak-Sneppen Model. First of all it has empty cells, whereas the original method did not have those. These cells are needed to migrate cells to a location where they might get a higher fitness. The interactions with the neighbours remain intact, but a neighbouring migrating cell results in a percentual decrease of the fitness instead of a randomised new fitness.

Each time step, the following steps are taken:
<ol type="A">
<li>Select the cell with the lowest fitness within the model.</li>
<li>Select a random empty cell and give this cell a new fitness using the formula mentioned below.</li>
<img src="https://latex.codecogs.com/gif.latex?new\_fitness&space;=&space;old\_fitness&space;&plus;&space;|\mathcal{N}(0,&space;\frac{1&space;-&space;old\_fitness}{3})|" class="latex"/>
<li>Empty the previously selected cell at A. and lower the cells within the <a href="https://en.wikipedia.org/wiki/Von_Neumann_neighborhood">von Neumann Neighbourhood</a> by a certain factor.</li>
</ol>

<img src="http://slum.life/images/bak-sneppen_expl.png" width="100%" class="no-border"/>

## Ages

When visualizing the simulation, the values plotted are the ages. These are defined as the number of timesteps that a person lives in a certain cell. This means that the age of a cell is incremented during each timestep that someone lives in certain cell and is set to zero when someone moves to another cell.

<div class='gif_container'>
<img class='gif' src="http://slum.life/videos/slum_barebones.png" width="100%"/>
<div class="overlay"></div>
<div class="play_button">&#9658;</div>
</div>
<span class="description">A simulation of the basic Slum Migration Model.</span>

## Avalanches

Just like in the [Bak-Tang-Wiesenfeld model](https://en.wikipedia.org/wiki/Abelian_sandpile_model) a single changing cell might induce a cascade of changing cells (avalanche). A single person moving away might inspire its neighbours to move away, and they again might inspire theirs, etc. To quantify this behaviour we measure the avalanche size. As an avalanche starts the fitness of the starting cell is set as limit for the avalanche. As long as the consecutive mutations are below this value its still part of the same avalanche, otherwise another started.

<img src="http://slum.life/images/avalanche_sizes.svg" width="100%"/>
## Influence of Slum Parameters

<img src="http://slum.life/images/emptypercent10x20000.svg" width="49%" class="no-border"/>
<img src="http://slum.life/images/slumsize20x25000.svg" width="49%" class="no-border"/>
<span class="description">The effect of the empty percentage of cells and slum size on the K of the powerlaw distribution of avalanche sizes. Each percentage was tested 10 times for 20000 time steps, each slum size 20 times for 25000 time steps.</span>

The slum size used and the empty percentage of cells within a slum often have negligible effects. Only when the empty percentage of cells becomes very large (95%), the variance of the K value becomes a lot higher than with lower precentages of empty cells. This can be explained by the fact that filled cells should have neighbours to cause avalanches. Before all cells are clustered it takes some time - which means that the warming up period can vary a lot, depending on the initial spread.  

Another effect can be seen with very small slum sizes. In this particular cases avalanches seem to encounter themselves through the periodic boundaries.

## More Slums
<img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Dharavi_India.jpg" width="100%"/>

<img src="http://slum.life/images/nrslums10x20000.svg" width="100%" class="no-border"/>
<span class="description">The effect of the number of slums on the K of the powerlaw distribution of avalanche sizes. The total number of cells within the simulation remained the same. Each size was tested 10 times for 20000 time steps.</span>

<div class='gif_container'>
<img class='gif' src="http://slum.life/videos/slum_multiple.png" width="100%"/>
<div class="overlay"></div>
<div class="play_button">&#9658;</div>
</div>
<span class="description"></span>

<div class='gif_container'>
<img class='gif' src="http://slum.life/videos/slum_network.png" width="100%"/>
<div class="overlay"></div>
<div class="play_button">&#9658;</div>
</div>
<span class="description"></span>

## The Destination

### The Optimal Location within a Slum
When moving to a new spot, people don't move to a random cell. People mostly move to spots where people already live. Therefor the probability of someone moving to a certain empty cell within a given slum is calculated as follows:

<img src="https://latex.codecogs.com/gif.latex?p&space;=&space;\frac{\&hash;neighbours^2&space;&plus;&space;0.01}{p_{total}}" class="latex"/>

### A Better Slum Selection Strategy
In real life, people won't move to a random slum. They will probably have a preference for slums where people are happier than they currently are. Instead of moving randomly, their strategy would move to a slum with happier people. A third more extreme moving strategy would be to pick the slum with the happiest people which has room for a new person.

The effects of these three different moving strategies on the K of the powerlaw distribution are shown in the figure below.

<img src="http://slum.life/images/strategy10x20000.svg" width="100%" class="no-border"/>
<span class="description">The effect of slum selection strategy on the K of the powerlaw distribution of avalanche sizes. Each strategy was tested 10 times for 20000 time steps.</span>

## Emergence of new slums

<div class='gif_container'>
<img class='gif' src="http://slum.life/videos/slum_new_slum.png" width="100%"/>
<div class="overlay"></div>
<div class="play_button">&#9658;</div>
</div>
<span class="description"></span>

<div class='gif_container'>
<img class='gif' src="http://slum.life/videos/slum_network_new.png" width="100%"/>
<div class="overlay"></div>
<div class="play_button">&#9658;</div>
</div>
<span class="description"></span>

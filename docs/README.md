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

<img src="http://slum.life/images/bak-sneppen_expl.png" width="100%"/>

<div class='gif_container'>
<img class='gif' src="http://slum.life/videos/slum_barebones.png" width="100%"/>
<div class="overlay"></div>
<div class="play_button">&#9658;</div>
</div>
<span class="description">A simulation of the basic Slum Migration Model.</span>

## Influence of Slum Parameters

<img src="http://slum.life/images/emptypercent10x20000.svg" width="50%" class="no-border"/>
<img src="http://slum.life/images/slumsize20x25000.svg" width="50%" class="no-border"/>


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
Avalanches .

Ages

## The Destination

### The Optimal Location within a Slum

### A Better Slum Selection Strategy
In real life, people won't move to a random slum. They will probably have a preference for slums where people are happier than they currently are. 

The effects of these three different moving strategies on the K of the powerlaw distribution are shown in the figure below.

<img src="http://slum.life/images/strategy10x20000.svg" width="100%" class="no-border"/>
<span class="description">The effect of slum selection strategy on the K of the powerlaw distribution of avalanche sizes. The total number of cells within the simulation remained the same. Each size was tested 10 times for 20000 time steps.</span>

## New Slum Locations





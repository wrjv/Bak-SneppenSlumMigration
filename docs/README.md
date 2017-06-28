<head>

 <script 
 
 src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js">
 </script>
 <script>
 
 $(document).ready(function()
 {
     $(".gif").hover(
         function()
         {
           var src = $(this).attr("src");
           $(this).attr("src", src.replace(/\.png$/i, ".gif"));
         },
         function()
         {
           var src = $(this).attr("src");
           $(this).attr("src", src.replace(/\.gif$/i, ".png"));
         });
 });

 document.getElementById("project_title").innerHTML = "Complex Systems Simulation";
</script>

</head>


<style>
 img {
 	margin: 0 auto;
 	display: block;
  	max-width: 2000px;
 }

 img.latex {
 	border: 0; 
 	outline: 0;
 	box-shadow: none;
 }

 #main_content, .inner {
 	max-width: 880px !important;
 }

 #project_title, #project_tagline {
 	text-align: center
 }
</style>

## A 2D Bak-Sneppen Slum Migration Model

Slum migration is a phenomenon mostly studied by social sciences and a topic difficult to quantify and make tangible. By modifying a 2D [Bak-Sneppen model](https://en.wikipedia.org/wiki/Bak%E2%80%93Sneppen_model), often used to model evolution, we attempt to generate the migration patterns. We show that by using only a few rules in a simple cellular automaton complex migration patterns start to occur. 

<img src="https://familyincluded.com/wp-content/uploads/2016/07/17252250461_11094da4cd_k.jpg" width="100%"/>

## The Basic Model

The basic version of the Slum Migration Model has only some slight modifications with respect to the basic 2D Bak-Sneppen Model. First of all it has empty cells, whereas the original method did not have those. These cells are needed to migrate cells to a location where they might get a higher fitness. The interactions with the neighbours remain intact, but a neighbouring migrating cell results in a percentual decrease of the fitness instead of a randomised new fitness.

Each time step, the following steps are taken:
<ol type="A">
<li>Select the cell with the lowest fitness within the model.</li>
<li>Select a random empty cell and give this cell a new fitness using the formula mentioned below.</li>
<img src="https://latex.codecogs.com/gif.latex?new\_fitness&space;=&space;old\_fitness&space;&plus;&space;|\mathcal{N}(0,&space;\frac{1&space;-&space;old\_fitness}{3})|" class="latex"/>
<li>Empty the previously selected cell at A. and lower the cells within the <a href="https://en.wikipedia.org/wiki/Von_Neumann_neighborhood">von Neumann Neighbourhood</a> by a certain factor.</li>
</ol>

<img src="http://slum.life/images/bak-sneppen_expl.png" width="100%" alt="Basic Bax-Sneppen steps."/>

<img class='gif' src="http://slum.life/videos/slum_barebones.png" width="100%" alt="An indian slum."/>

## Influence of Slum Parameters

## More Slums
<img src="https://static01.nyt.com/images/2013/09/12/world/asia/12-slum-mumbai-indiaInk/12-slum-mumbai-indiaInk-superJumbo.jpg" width="100%"/>


## The Destination

### The Optimal Location within a Slum

### The Optimal Slum

## New Slum Locations





# Team Information

**Course:** _[fill your course code and name]_

**Semester:** Semester 2, 2020

**Instructor:** _[name of your instructor]_

**Team name:** Neural Nom Nom

**Team members:**

249845 - Markus Baier - markus.baier01@estudiant.upf.edu - Secras
249844 - Oliver Ingvarsson - oliver.ingvarsson01@estudiant.upf.edu - OliverO15
* Student 1's Student number - Full Name - Student email - Student Github id

Replace the lines above with the correct details of members. Delete or add lines as needed.

Student numbers should just be the **numbers**.

# Delete Later

# Strategy Ideas

**Defensive**

When he **doesn't** know where PacMan is:
- Can camp at the center by the edge of blue/red to protect the edge to try to get the PacMan while he gets out or tries to come in
- Camp where the most food clusters reside

When he **knows** where pacman is:
- MinMax to predict what route the PacMan will go and catch him that way
- Find an algorithm to corner the PacMan
- Learn from his actions. And use expectminimax (last resort)

When ghosts get scared:
- Call attacker in when defensive get's scared (if he doesn't get scared himself)
- Make the defensive agent go in attack.

**Offensive**

How to find food?
- (DONE) Try to find food clusters, first near the edge.
- (DONE) Don't get too greedy, how far is it to the next food cluster and how far is it back?
- (DONE) When he eats a power capsule he gets greedy and is allowed further back, until he eats a ghost, then the timer gets reduced on the amount of team he has left to get back to base.

When to go back?
- (DONE) If we have the min amount of food then go back over the edge.
- (DONE) If we don't have any food then try to find food and run away from the ghosts but still inside the enemy area.
- Avoid getting stuck in a corner when running away from ghosts.

Completed Strategy:
- We rank the food clusters based on their size (how many food items there are in a cluster), and also their location (how far they are inside the enemy base)
- We start by going to the biggest food cluster that is closest to the edge of the enemy base
- If PacMan eats 4 food items in a row, we go back to our base with the food
- If we are in the enemy base, and we see a ghost, we run away from the ghost (doesn't matter to where now)
- If capsule is eaten, we proiritize finding the biggest food clusters and remove the max 4 food eaten rule and avoiding ghosts.
- Minimax (not with alpha/beta pruning) implemented for escaping ghosts
  - We predict that the ghost will move towards us, and we move away from the ghost
  - If food eaten is less than 2 then we will not prioritize running to the base, only avoid the ghost

TODO:
- Try alpha/beta pruning for minimax to be able to increase the depth of the search
- Food cluster selection
  - Being able to stick to one cluster at a time so it can be switched if something happens
  - Or for selecting random cluster to decrease predictability
- Turn into defensive agent when food from our team is low

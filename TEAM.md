# Team Information

**Course:** _[fill your course code and name]_

**Semester:** Semester 2, 2020

**Instructor:** _[name of your instructor]_

**Team name:** _[replace this with team name in plan text]_

**Team members:**

* Student 1's Student number - Full Name - Student email - Student Github id
* Student 2's Student number - Full Name - Student email - Student Github id
* Student 3's Student number - Full Name - Student email - Student Github id

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
- Don't get too greedy, how far is it to the next food cluster and how far is it back?
- When he eats a power capsule he gets greedy and is allowed further back, until he eats a ghost, then the timer gets reduced on the amount of team he has left to get back to base.

When to go back?
- (DONE) If we have the min amount of food then go back over the edge.
- (DONE) If we don't have any food then try to find food and run away from the ghosts but still inside the enemy area.
- Avoid getting stuck in a corner when running away from ghosts.

Completed Strategy:
- We rank the food clusters based on their size (how many food items there are in a cluster), and also their location (how far they are inside the enemy base)
- We start by going to the biggest food cluster that is closest to the edge of the enemy base
- If PacMan eats 4 food items in a row, we go back to our base with the food
- If we are in the enemy base, and we see a ghost, we run away from the ghost (doesn't matter to where now)

TODO:
- Big thing is to find an algorithm to rank or make sure the PacMan when fleeing from a ghost doesn't go into a corner where he can't get out.
  - Or basically make a better escape plan when ghost is nearby

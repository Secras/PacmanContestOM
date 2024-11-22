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
- Don't get too greedy, how far is it to the next food cluster and how far is it back?
- Try to find food first near the edge 
- When he eats a power capsule he gets greedy and is allowed further back, until he eats a ghost, then the timer gets reduced on the amount of team he has left to get back to base.

When to go back?
- If we have the min amount of food then go back over the edge.
- If we don't have any food then try to find food and run away from the ghosts but still inside the enemy area.
- 

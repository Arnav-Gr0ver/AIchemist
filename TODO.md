
workflow diagram

user wants to do a blind replication study of a paper

system is handed a paper / i.e. it's uploaded, or there can be a search function later on (add this later, not part of MVP)

system takes paper then decomposes it into parts

decomposed paper is refined and filtered for only task + eval relevant pieces (like you dont need an intro to do a replication study)

system comes up with set of steps + task requirements in order to achieve results

- also determines training requirements given sandbox, I.E. is it feasible in given sandbox

1. sandbox setup
    * dependencies installed
    * ...
2. web retrieval (if necessary)
    * models retrieved
    * datasets retrieved
3. architecture setup
    * stuff setup
4. training setup
    * script setup
5. training + logging
6. eval
7. ablation studies

in future user can backtrack to any step
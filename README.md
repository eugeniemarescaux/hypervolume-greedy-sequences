# hypervolume-greedy-sequences
This project contains code to compute a sequence of objective-vectors which greedily maximizes the hypervolume.
Seven benchmark Pareto fronts are available. 

To compute the 1000-th first terms of a greedy sequence for these seven Pareto fronts and the reference point r=[1,1], one could use the following bash function:
```
function expes_greedy() {
  for fun in convex-biL convex-doublesphere convex-zdt1 concave-biL concave-dtlz2 concave-zdt2 linear
    do
      pretty_exp "python -m script-greedy-computation --nb_runs=3 --fun=$fun --p=1000" $fun
    done
}
```

This code has not been properly tested with other reference points. 

## Parallel Graph Search (A Case Study)
This repo contains a case study for parallel graph search algorithms, specifically Bellman Ford, Dijkstra and A-star algorithms. 

### Compilation
This repo uses (GNU) `make` buildsystem and `nvcc` CUDA compiler in order to compile and link the source files. Also, you might want to edit the `Makefile` and change the `-arch sm_72` flag according to your GPU's compute capability. 
In order to compile, link and run, just run:  
```
make all
./pg_search
```

### Write-Up
You can find a detailed write-up about the code and performance evaluation at my website: [https://khizar-anjum.github.io/projects/pgsearch.html](https://khizar-anjum.github.io/projects/pgsearch.html)

### References used
We used the following references in our implementation:  
- Yichao Zhou and Jianyang Zeng. 2015. Massively parallel a* search on a GPU. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI'15). AAAI Press, 1248?1254.
- [9th DIMACS Implementation Challenge - Shortest Paths](http://users.diag.uniroma1.it/challenge9/download.shtml)
- [rapidcsv by d99kris](https://github.com/d99kris/rapidcsv)
- Leskovec, Jure, and Andrej Krevl. "SNAP datasets: stanford large network dataset collection; 2014." URL http://snap. stanford. edu/data (2016): 49.

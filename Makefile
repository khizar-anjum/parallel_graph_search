SRC = src
INC = include
FLAGS = -I$(INC) -arch sm_75
ifeq ($(wildcard /opt/cuda/bin/nvcc),)
	NVCC=nvcc
else
	NVCC=/opt/cuda/bin/nvcc
endif

ODIR=obj

LIBS=

_OBJ = astar_cpu.o bellmanford_cpu.o dijkstra_cpu.o graph_cpu.o pqueue_cpu.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_OBJ_CU = astar_gpu.o bellmanford_gpu.o dijkstra_gpu.o pqueue_gpu.o
OBJ_CU = $(patsubst %,$(ODIR)/%, $(_OBJ_CU))

directories:
	mkdir -p $(ODIR)
	mkdir -p results

$(ODIR)/%_cpu.o: $(SRC)/%.cpp
	$(NVCC) --device-c -o $@ $^ $(FLAGS)

$(ODIR)/%_gpu.o: $(SRC)/%.cu
	$(NVCC) --device-c -o $@ $^ $(FLAGS)

pg_search: main.cpp $(OBJ) $(OBJ_CU)
	$(NVCC) -o $@ $^ $(FLAGS) $(LIBS)

all: directories pg_search

.PHONY: clean directories

clean:
	rm -f $(ODIR)/*.o
	rm pg_search

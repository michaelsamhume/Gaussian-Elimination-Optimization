CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
TARGETS = baseline transposed tiled openmp neon

all: $(addprefix bin/,$(TARGETS))

bin:
	mkdir -p bin

bin/baseline: src/baseline.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/transposed: src/transposed.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/tiled: src/tiled.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/openmp: src/openmp.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

bin/neon: src/neon.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

benchmark: all
	bash benchmarks/run.sh

clean:
	rm -rf bin

.PHONY: all benchmark clean

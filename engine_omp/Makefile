# ── BRAHMS CPU Engine Makefile ─────────────────────────────────────────────────
#
# Usage:
#   make                            # default grid
#   make NX=64 NY=64 NZ=100 NT=256
#   make NX=8 NY=8 NZ=20 NT=8      # quick test
#   make PRECISION=double          # double-precision build (default: float)
#   make clean
#
# Switching PRECISION or the grid size triggers a rebuild automatically (the
# settings are recorded in .build_cfg).
#
# Dependencies: g++ ≥ 9, OpenMP, HDF5, nlohmann/json, and FFTW3 in the chosen
#               precision (libfftw3f for float, libfftw3 for double; both with
#               their _omp variants)
# ──────────────────────────────────────────────────────────────────────────────

NX  ?= 128
NY  ?= 128
NZ  ?= 100
NT  ?= 512

# float (default) or double. Selects the field type, FFTW libraries and HDF5
# output type (see headers/DataTypes_cpu.hpp).
PRECISION ?= float
ifeq ($(PRECISION),double)
    PREC_FLAGS  := -DBRAHMS_DOUBLE
    FFTW_SUFFIX :=
else ifeq ($(PRECISION),float)
    PREC_FLAGS  :=
    FFTW_SUFFIX := f
else
    $(error PRECISION must be 'float' or 'double' (got '$(PRECISION)'))
endif

ifeq ($(OS),Windows_NT)
    WINDOWS := 1
endif

# ── Paths ─────────────────────────────────────────────────────────────────────
ifdef WINDOWS
# MSYS2 MinGW-w64 prefix (pacman -S mingw-w64-x86_64-{gcc,make,fftw,hdf5,nlohmann-json}).
# Override with MINGW_PREFIX=... if MSYS2 lives elsewhere.
MINGW_PREFIX ?= C:/msys64/mingw64
FFTW3_INC := $(MINGW_PREFIX)/include
FFTW3_LIB := $(MINGW_PREFIX)/lib
HDF5_INC  ?= $(MINGW_PREFIX)/include
HDF5_LIB  ?= $(MINGW_PREFIX)/lib
JSON_INC  ?= $(MINGW_PREFIX)/include
else
# FFTW3 — try common conda install locations (any user), then pkg-config,
# then the standard system paths (Debian/Ubuntu multiarch layout).
CONDA_FFTW := $(shell find $(HOME)/anaconda3/pkgs $(HOME)/miniconda3/pkgs \
                /opt/conda/pkgs -maxdepth 1 -type d -name "fftw-*" \
                2>/dev/null | sort -r | head -1)
PKGCONFIG_FFTW_LIB := $(shell pkg-config --variable=libdir fftw3$(FFTW_SUFFIX) 2>/dev/null)
MULTIARCH := $(shell dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null)

ifneq ($(CONDA_FFTW),)
  FFTW3_INC := $(CONDA_FFTW)/include
  FFTW3_LIB := $(CONDA_FFTW)/lib
else ifneq ($(PKGCONFIG_FFTW_LIB),)
  FFTW3_INC := $(shell pkg-config --variable=includedir fftw3$(FFTW_SUFFIX))
  FFTW3_LIB := $(PKGCONFIG_FFTW_LIB)
else ifneq ($(MULTIARCH),)
  FFTW3_INC := /usr/include
  FFTW3_LIB := /usr/lib/$(MULTIARCH)
else
  FFTW3_INC := /usr/include
  FFTW3_LIB := /usr/lib
endif

HDF5_INC  ?= /usr/include/hdf5/serial
HDF5_LIB  ?= /usr/lib/x86_64-linux-gnu/hdf5/serial
JSON_INC  ?= /usr/include
endif

# ── Compiler ──────────────────────────────────────────────────────────────────
CXX = g++

GRID_FLAGS = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DNT=$(NT)

CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp \
           $(GRID_FLAGS) $(PREC_FLAGS) \
           -I$(FFTW3_INC) \
           -I$(HDF5_INC) \
           -I$(JSON_INC)

LDFLAGS = -fopenmp \
          -L$(FFTW3_LIB) -lfftw3$(FFTW_SUFFIX)_omp -lfftw3$(FFTW_SUFFIX) \
          -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 \
          -lm
ifndef WINDOWS
LDFLAGS += -Wl,-rpath,$(FFTW3_LIB)
endif

ifdef WINDOWS
TARGET = twm_cpu.exe
else
TARGET = twm_cpu
endif
SRCS   = twm_cpu.cpp

# make does not notice that -D flags changed, so record them in a stamp file and
# force a rebuild (via the always-out-of-date FORCE target, which does not
# depend on file timestamps) whenever they differ from the last build.
# Needs GNU make >= 4.2 for $(file <...); older versions just skip the check.
BUILD_CFG := PRECISION=$(PRECISION) NX=$(NX) NY=$(NY) NZ=$(NZ) NT=$(NT)
STAMP     := .build_cfg
REBUILD   :=
ifeq ($(filter 3.% 4.0% 4.1%,$(MAKE_VERSION)),)
ifneq ($(file <$(STAMP)),$(BUILD_CFG))
$(file >$(STAMP),$(BUILD_CFG))
REBUILD := FORCE
endif
endif

.PHONY: all clean info FORCE

all: $(TARGET)

FORCE:

$(TARGET): $(SRCS) headers/*.hpp $(REBUILD)
	@echo "  [CXX] Compiling BRAHMS CPU engine..."
	@echo "        Grid: NX=$(NX)  NY=$(NY)  NZ=$(NZ)  NT=$(NT)  Precision: $(PRECISION)"
	@echo "        FFTW3: $(FFTW3_LIB)"
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "  [OK]  ./$(TARGET) ready"

clean:
ifdef WINDOWS
	-del /Q $(TARGET) *.o $(STAMP) 2>NUL
else
	rm -f $(TARGET) *.o $(STAMP)
endif

info:
	@echo "NX=$(NX) NY=$(NY) NZ=$(NZ) NT=$(NT)"
	@echo "FFTW3_INC=$(FFTW3_INC)"
	@$(CXX) --version

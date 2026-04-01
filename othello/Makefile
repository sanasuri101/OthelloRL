CC      ?= gcc
PYTHON  ?= python3

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    ARCH_FLAG = -arch $(shell $(PYTHON) -c "import platform; print(platform.machine())")
    LDFLAGS    = -undefined dynamic_lookup
else
    ARCH_FLAG =
    LDFLAGS    =
endif

CFLAGS       = -O3 -shared -fPIC -Wall -Wextra -std=c11 $(ARCH_FLAG)
RAYLIB_FLAGS = $(shell pkg-config --cflags --libs raylib 2>/dev/null || echo "-lraylib")
PYTHON_FLAGS = $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
NUMPY_FLAGS  = $(shell $(PYTHON) -c "import numpy; print('-I' + numpy.get_include())")
TARGET       = binding$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: all clean test

all: $(TARGET)

$(TARGET): binding.c othello.h negamax.h render.h
	$(CC) $(CFLAGS) $(LDFLAGS) $(PYTHON_FLAGS) $(NUMPY_FLAGS) $(RAYLIB_FLAGS) -o $@ binding.c

clean:
	rm -f $(TARGET) *.o

test:
	$(PYTHON) -m pytest tests/ -v

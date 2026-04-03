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

CFLAGS    = -O3 -shared -fPIC -Wall -Wextra -std=c11 $(ARCH_FLAG)
NO_RENDER ?= 0
ifeq ($(NO_RENDER), 1)
    RAYLIB_FLAGS = -DNO_RENDER
else
    RAYLIB_FLAGS = $(shell PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig:/usr/local/lib/pkgconfig pkg-config --cflags --libs raylib 2>/dev/null || echo "-lraylib")
endif
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

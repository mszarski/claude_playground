# Gilgamesh 64 - N64 build (libdragon preview + tiny3d)
#   export N64_INST=/opt/libdragon          # libdragon toolchain
#   make T3D_INST=/path/to/tiny3d            # tiny3d checkout (built with its ./build.sh)
BUILD_DIR = build
T3D_INST ?= $(HOME)/tiny3d

include $(N64_INST)/include/n64.mk
include $(T3D_INST)/t3d.mk

N64_CFLAGS += -std=gnu2x -O2 -Wno-error=unused-variable -Wno-error=unused-function $(EXTRA_CFLAGS)

PROJECT_NAME = gilgamesh64
src = $(wildcard src/*.c)

assets_conv = filesystem/monogram.font64 filesystem/abaddon.font64

all: $(PROJECT_NAME).z64

filesystem/monogram.font64: assets/monogram.ttf
	@mkdir -p $(dir $@)
	@echo "    [FONT] $@"
	$(N64_MKFONT) $(MKFONT_FLAGS) --size 16 --outline 1 -o filesystem "$<"

filesystem/abaddon.font64: assets/abaddon.ttf
	@mkdir -p $(dir $@)
	@echo "    [FONT] $@"
	$(N64_MKFONT) $(MKFONT_FLAGS) --size 16 --outline 1 -o filesystem "$<"

$(BUILD_DIR)/$(PROJECT_NAME).dfs: $(assets_conv)
$(BUILD_DIR)/$(PROJECT_NAME).elf: $(src:%.c=$(BUILD_DIR)/%.o)

$(PROJECT_NAME).z64: N64_ROM_TITLE = "GILGAMESH 64"
$(PROJECT_NAME).z64: $(BUILD_DIR)/$(PROJECT_NAME).dfs

clean:
	rm -rf $(BUILD_DIR) filesystem *.z64

-include $(wildcard $(BUILD_DIR)/src/*.d)

.PHONY: all clean

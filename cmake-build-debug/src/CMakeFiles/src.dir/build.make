# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/pedro/clion-2017/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/pedro/clion-2017/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug

# Include any dependencies generated for this target.
include src/CMakeFiles/src.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/src.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/src.dir/flags.make

src/CMakeFiles/src.dir/processing.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/processing.cpp.o: ../src/processing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/src.dir/processing.cpp.o"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/src.dir/processing.cpp.o -c /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src/processing.cpp

src/CMakeFiles/src.dir/processing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/processing.cpp.i"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src/processing.cpp > CMakeFiles/src.dir/processing.cpp.i

src/CMakeFiles/src.dir/processing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/processing.cpp.s"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src/processing.cpp -o CMakeFiles/src.dir/processing.cpp.s

src/CMakeFiles/src.dir/processing.cpp.o.requires:

.PHONY : src/CMakeFiles/src.dir/processing.cpp.o.requires

src/CMakeFiles/src.dir/processing.cpp.o.provides: src/CMakeFiles/src.dir/processing.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/src.dir/build.make src/CMakeFiles/src.dir/processing.cpp.o.provides.build
.PHONY : src/CMakeFiles/src.dir/processing.cpp.o.provides

src/CMakeFiles/src.dir/processing.cpp.o.provides.build: src/CMakeFiles/src.dir/processing.cpp.o


src/CMakeFiles/src.dir/utility.cpp.o: src/CMakeFiles/src.dir/flags.make
src/CMakeFiles/src.dir/utility.cpp.o: ../src/utility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/src.dir/utility.cpp.o"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/src.dir/utility.cpp.o -c /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src/utility.cpp

src/CMakeFiles/src.dir/utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/src.dir/utility.cpp.i"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src/utility.cpp > CMakeFiles/src.dir/utility.cpp.i

src/CMakeFiles/src.dir/utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/src.dir/utility.cpp.s"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src/utility.cpp -o CMakeFiles/src.dir/utility.cpp.s

src/CMakeFiles/src.dir/utility.cpp.o.requires:

.PHONY : src/CMakeFiles/src.dir/utility.cpp.o.requires

src/CMakeFiles/src.dir/utility.cpp.o.provides: src/CMakeFiles/src.dir/utility.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/src.dir/build.make src/CMakeFiles/src.dir/utility.cpp.o.provides.build
.PHONY : src/CMakeFiles/src.dir/utility.cpp.o.provides

src/CMakeFiles/src.dir/utility.cpp.o.provides.build: src/CMakeFiles/src.dir/utility.cpp.o


# Object files for target src
src_OBJECTS = \
"CMakeFiles/src.dir/processing.cpp.o" \
"CMakeFiles/src.dir/utility.cpp.o"

# External object files for target src
src_EXTERNAL_OBJECTS =

src/libsrc.a: src/CMakeFiles/src.dir/processing.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/utility.cpp.o
src/libsrc.a: src/CMakeFiles/src.dir/build.make
src/libsrc.a: src/CMakeFiles/src.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libsrc.a"
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && $(CMAKE_COMMAND) -P CMakeFiles/src.dir/cmake_clean_target.cmake
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/src.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/src.dir/build: src/libsrc.a

.PHONY : src/CMakeFiles/src.dir/build

src/CMakeFiles/src.dir/requires: src/CMakeFiles/src.dir/processing.cpp.o.requires
src/CMakeFiles/src.dir/requires: src/CMakeFiles/src.dir/utility.cpp.o.requires

.PHONY : src/CMakeFiles/src.dir/requires

src/CMakeFiles/src.dir/clean:
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src && $(CMAKE_COMMAND) -P CMakeFiles/src.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/src.dir/clean

src/CMakeFiles/src.dir/depend:
	cd /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/src /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src /home/pedro/Documents/of_v0.9.8_linux64_release/apps/myApps/extractor/cmake-build-debug/src/CMakeFiles/src.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/src.dir/depend


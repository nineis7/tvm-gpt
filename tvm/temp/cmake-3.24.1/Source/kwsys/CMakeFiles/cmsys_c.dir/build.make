# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Bootstrap.cmk/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1

# Include any dependencies generated for this target.
include Source/kwsys/CMakeFiles/cmsys_c.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/kwsys/CMakeFiles/cmsys_c.dir/progress.make

# Include the compile flags for this target's objects.
include Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make

Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o: Source/kwsys/ProcessUNIX.c
Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_C_HAS_PTRDIFF_T=1 -DKWSYS_C_HAS_SSIZE_T=1 -DKWSYS_C_HAS_CLOCK_GETTIME_MONOTONIC=1 -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o -MF CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o.d -o CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/ProcessUNIX.c

Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/ProcessUNIX.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_C_HAS_PTRDIFF_T=1 -DKWSYS_C_HAS_SSIZE_T=1 -DKWSYS_C_HAS_CLOCK_GETTIME_MONOTONIC=1 -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/ProcessUNIX.c > CMakeFiles/cmsys_c.dir/ProcessUNIX.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/ProcessUNIX.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_C_HAS_PTRDIFF_T=1 -DKWSYS_C_HAS_SSIZE_T=1 -DKWSYS_C_HAS_CLOCK_GETTIME_MONOTONIC=1 -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/ProcessUNIX.c -o CMakeFiles/cmsys_c.dir/ProcessUNIX.c.s

Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.o: Source/kwsys/Base64.c
Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.o -MF CMakeFiles/cmsys_c.dir/Base64.c.o.d -o CMakeFiles/cmsys_c.dir/Base64.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/Base64.c

Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/Base64.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/Base64.c > CMakeFiles/cmsys_c.dir/Base64.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/Base64.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/Base64.c -o CMakeFiles/cmsys_c.dir/Base64.c.s

Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.o: Source/kwsys/EncodingC.c
Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) -DKWSYS_ENCODING_DEFAULT_CODEPAGE=CP_UTF8 $(C_INCLUDES) $(C_FLAGS) -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.o -MF CMakeFiles/cmsys_c.dir/EncodingC.c.o.d -o CMakeFiles/cmsys_c.dir/EncodingC.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/EncodingC.c

Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/EncodingC.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) -DKWSYS_ENCODING_DEFAULT_CODEPAGE=CP_UTF8 $(C_INCLUDES) $(C_FLAGS) -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/EncodingC.c > CMakeFiles/cmsys_c.dir/EncodingC.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/EncodingC.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) -DKWSYS_ENCODING_DEFAULT_CODEPAGE=CP_UTF8 $(C_INCLUDES) $(C_FLAGS) -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/EncodingC.c -o CMakeFiles/cmsys_c.dir/EncodingC.c.s

Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.o: Source/kwsys/MD5.c
Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.o -MF CMakeFiles/cmsys_c.dir/MD5.c.o.d -o CMakeFiles/cmsys_c.dir/MD5.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/MD5.c

Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/MD5.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/MD5.c > CMakeFiles/cmsys_c.dir/MD5.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/MD5.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/MD5.c -o CMakeFiles/cmsys_c.dir/MD5.c.s

Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.o: Source/kwsys/Terminal.c
Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.o -MF CMakeFiles/cmsys_c.dir/Terminal.c.o.d -o CMakeFiles/cmsys_c.dir/Terminal.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/Terminal.c

Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/Terminal.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/Terminal.c > CMakeFiles/cmsys_c.dir/Terminal.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/Terminal.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/Terminal.c -o CMakeFiles/cmsys_c.dir/Terminal.c.s

Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.o: Source/kwsys/System.c
Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_C_HAS_PTRDIFF_T=1 -DKWSYS_C_HAS_SSIZE_T=1 -DKWSYS_C_HAS_CLOCK_GETTIME_MONOTONIC=1 -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.o -MF CMakeFiles/cmsys_c.dir/System.c.o.d -o CMakeFiles/cmsys_c.dir/System.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/System.c

Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/System.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_C_HAS_PTRDIFF_T=1 -DKWSYS_C_HAS_SSIZE_T=1 -DKWSYS_C_HAS_CLOCK_GETTIME_MONOTONIC=1 -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/System.c > CMakeFiles/cmsys_c.dir/System.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/System.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_C_HAS_PTRDIFF_T=1 -DKWSYS_C_HAS_SSIZE_T=1 -DKWSYS_C_HAS_CLOCK_GETTIME_MONOTONIC=1 -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/System.c -o CMakeFiles/cmsys_c.dir/System.c.s

Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/flags.make
Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.o: Source/kwsys/String.c
Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.o: Source/kwsys/CMakeFiles/cmsys_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.o"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_STRING_C -MD -MT Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.o -MF CMakeFiles/cmsys_c.dir/String.c.o.d -o CMakeFiles/cmsys_c.dir/String.c.o -c /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/String.c

Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmsys_c.dir/String.c.i"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_STRING_C -E /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/String.c > CMakeFiles/cmsys_c.dir/String.c.i

Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmsys_c.dir/String.c.s"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -DKWSYS_STRING_C -S /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/String.c -o CMakeFiles/cmsys_c.dir/String.c.s

# Object files for target cmsys_c
cmsys_c_OBJECTS = \
"CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o" \
"CMakeFiles/cmsys_c.dir/Base64.c.o" \
"CMakeFiles/cmsys_c.dir/EncodingC.c.o" \
"CMakeFiles/cmsys_c.dir/MD5.c.o" \
"CMakeFiles/cmsys_c.dir/Terminal.c.o" \
"CMakeFiles/cmsys_c.dir/System.c.o" \
"CMakeFiles/cmsys_c.dir/String.c.o"

# External object files for target cmsys_c
cmsys_c_EXTERNAL_OBJECTS =

Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/ProcessUNIX.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/Base64.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/EncodingC.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/MD5.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/Terminal.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/System.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/String.c.o
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/build.make
Source/kwsys/libcmsys_c.a: Source/kwsys/CMakeFiles/cmsys_c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking C static library libcmsys_c.a"
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && $(CMAKE_COMMAND) -P CMakeFiles/cmsys_c.dir/cmake_clean_target.cmake
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmsys_c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Source/kwsys/CMakeFiles/cmsys_c.dir/build: Source/kwsys/libcmsys_c.a
.PHONY : Source/kwsys/CMakeFiles/cmsys_c.dir/build

Source/kwsys/CMakeFiles/cmsys_c.dir/clean:
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys && $(CMAKE_COMMAND) -P CMakeFiles/cmsys_c.dir/cmake_clean.cmake
.PHONY : Source/kwsys/CMakeFiles/cmsys_c.dir/clean

Source/kwsys/CMakeFiles/cmsys_c.dir/depend:
	cd /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1 /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1 /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys /home/nineis/ws/gpt-frontend/tvm/temp/cmake-3.24.1/Source/kwsys/CMakeFiles/cmsys_c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Source/kwsys/CMakeFiles/cmsys_c.dir/depend


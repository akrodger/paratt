################################################################################
##                                                                            ##
##                     build system for mpi-attac                             ##
##                       Written by Bram Rodgers                              ##
##                 Initial draft written June 29, 2022                        ##
##                                                                            ##
################################################################################

#A default compiler. (DC = Default Compiler)
DC=mpicc
#Program compilation flags (CFLAGS = Compiler Flags)
CFLAGS=-fpic -O3# -fopenmp #-g -Wall#
#Some default common linked libraries (CLINX = Common Linkages)
CLINX=#-lm -llapack 
#A default OpenMP compilation flag. Set blank for serial code.
OMP=-fopenmp

#An archiving software for making static libraries
AR=ar
#Flags for the archiving software
ARFLAGS=-rcs

#The file type extension of the source code used. (SRC_T = source type)
SRC_T=c
#A default file extension for executable files (EXE_T = executable type)
EXE_T=x

#A listing of source code. Automatically detects all in this folder
SRC_FILES=$(wildcard *.$(SRC_T))
#Dependencies listing for the source files
DEP_FILES=$(subst .$(SRC_T),.d,$(SRC_FILES))

#A listing of object files. Automatically generated based on SRC_FILES
OBJ_FILES=$(subst .$(SRC_T),.o,$(SRC_FILES))
#A name for an archive file constructured from the objs of this folder
ARCH_FILE="./libparatt.a"#$(shell basename $(CURDIR)).a
#A default name of an executable
EXE_NM=exec.$(EXE_T)

#Name of a folder contained within this folder.
SUBDIR=mpi_attac_src

#Begin Makefile recipes template
#.PHONY means that this recipe does not make a file which has the
#same name as this recipe. Example: the ``clean'' routine does not make
#a file called ``clean'' , instead it removes files.

#Default recipe of makefile.
.PHONY: default
default: arch
	@echo "Make for parallel attac helper functions."

#A recipe for renaming the file extension type of choice with .o files
#This is not ``.PHONY'' because it makes actual files with those names.
%.o : %.$(SRC_T)
	$(DC) $(CFLAGS) $(CFLAGS_EXTRA) $(OMP) -c $< -o $@
	
#Object file make recipe. Just calls all the %.o make rules.
.PHONY: objs
objs: $(OBJ_FILES)

#Including the dependent files. forces makefile to run deps every time.
-include deps

#Dependent files make recipe. Makes a .d file listing the includes of sources.
%.d: %.$(SRC_T)
	$(DC) $< -MM -MT $(@:.d=.o) > $@

#Object file make recipe. Just calls all the %.d make rules.
.PHONY: deps
deps: $(DEP_FILES)

#Creates an executable based on the EXE_NM variable listed above.
.PHONY: exec
exec: $(EXE_NM)

#Creates an executable binary file with the name stored in $(EXE_NM)
$(EXE_NM): $(OBJ_FILES)
	$(DC) $(CLINX) $(OMP) $^ -o $(EXE_NM)

#Creates an archived static library using the object files in this directory.
.PHONY: arch
arch: $(ARCH_FILE) libmpi_attac libserial_attac

#Rule for creating an archive file with name $(ARCH_FILE)
$(ARCH_FILE): $(OBJ_FILES)
	$(AR) $(ARFLAGS) $@ $^

#Go into subdirectory $(SUBDIR) and call the makefile contained there.
.PHONY: libmpi_attac
libmpi_attac:
	$(MAKE) -C $(SUBDIR) libmpi_attac
	
.PHONY: libserial_attac
libserial_attac:
	$(MAKE) -C $(SUBDIR) libserial_attac


#A basic recipe for cleaning up object files and executables.
.PHONY: clean
clean: cleansubdir
	@if rm *.o ; then echo "Removed object files."; fi
	@if rm *.a ; then echo "Removed archive files."; fi
	@if rm *.d ; then echo "Removed dependency files."; fi
	@if rm *.$(EXE_T) ; then echo "Removed executable files."; fi

#Calls the clean recipe of the subdirectory $(SUBDIR)
#Small projects do no need this. Simply uncomment to enable this.
.PHONY: cleansubdir
cleansubdir: 
	$(MAKE) -C $(SUBDIR) clean

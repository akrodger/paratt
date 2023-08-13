################################################################################
##                                                                            ##
##                   PARATT executables build system                          ##
##                       Written by Bram Rodgers                              ##
##                  Initial draft written July 3, 2022                        ##
##                                                                            ##
################################################################################

#A default compiler. (DC = Default Compiler)
DC=mpicc
#Program compilation flags (CFLAGS = Compiler Flags)
CFLAGS=-fpic -O3# -fopenmp#-g#
BLAS=-lblas
LAPACK=-llapack
FORTRAN=gfortran
#Some default common linked libraries (CLINX = Common Linkages)
CLINX=-L./source/ -L./source/mpi_attac_src/ -lparatt\
       -lserial_attac -lmpi_attac -lm -l$(FORTRAN) $(LAPACK) $(BLAS)
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
#All elecutatble names
EXE_TEST_FILES=$(subst .$(SRC_T),.$(EXE_T),$(SRC_FILES))

#name of an executable for testing memory allocation.
alloc_exe_nm=mpi_attac_alloc_test.$(EXE_T)
#Name of an executable for testing initial condition generation.
ic_exe_nm=mpi_attac_ic_test.$(EXE_T)

arch_file_path=./source/libparatt.a\
                ./source/mpi_attac_src/libmpi_attac.a\
                ./source/mpi_attac_src/libserial_attac.a\
                ./source/plm_1s/libplm_1d.a


#Name of a folder contained within this folder.
SUBDIR=source

#Begin Makefile recipes template
#.PHONY means that this recipe does not make a file which has the
#same name as this recipe. Example: the ``clean'' routine does not make
#a file called ``clean'' , instead it removes files.

#Default recipe of makefile.
.PHONY: default
default: exec

#make recipe for debug mode compile
.PHONY: debug_tests
debug_tests: CFLAGS_EXTRA=-g -Wall
debug_tests: tests

#make recipe for optimized native mode compile
.PHONY: opthigh_tests
opthigh_tests: CFLAGS_EXTRA=-Ofast
opthigh_tests: clean tests

#Creates an executable based on the EXE_NM variable listed above.
.PHONY: exec
exec:  $(EXE_TEST_FILES)

#Creates an executable binary file with the name stored in $(EXE_NM)
%.x: %.c libsubdir
	$(DC) $(CFLAGS) $(CFLAGS_EXTRA) $(OMP) $< -o $@ $(CLINX) 


#Go into subdirectory $(SUBDIR) and call the makefile contained there.
.PHONY: libsubdir
libsubdir:
	$(MAKE) -C $(SUBDIR) arch

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

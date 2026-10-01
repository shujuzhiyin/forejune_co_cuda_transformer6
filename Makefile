#~/pulp_box/forejune_co_cuda/transformer6/Makefile
PROG	= testMain
#source codes
SRCS =  $(PROG).cpp
#substitute .cpp by .o to obtain object filenames
OBJS = $(SRCS:.cpp=.o) util.o  transformer.o
#$< evaluates to the target's dependencies, 
#$@ evaluates to the target
$(PROG): $(OBJS)
	g++ -o $@ $(OBJS)  
$(OBJS): 
	g++ -c  -std=c++20 $*.cpp 
clean:
	rm $(OBJS) $(PROG)

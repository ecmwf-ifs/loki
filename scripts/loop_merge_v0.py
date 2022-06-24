#routine to merge loops with identical bounds

from loki import (Sourcefile,FindNodes,Loop,Transformer,Comment,as_tuple)
from loki.tools import flatten
from loki.transform.transform_loop import get_nested_loops

#class nested:
#      def __init__(self,stat,parent):
#          self.stat = stat
#          self.parent = parent


#read-in sourcefile and create subroutine object
source = Sourcefile.from_file('testing/src/test.F90')
routine = source['test']

#using visitor to locate all loops
loops = FindNodes(Loop).visit(routine.body)
nloops = len(loops)

#determining how many types of loop exist
#loops are considered to be of same type if variable and bounds (including step) match
nloop_types = 0
loop_type = []
for i in range(nloops):
    stat = False
    for j in range(i):
        if loops[i].variable == loops[j].variable and loops[i].bounds == loops[j].bounds:
            stat = True
            loop_type.append(loop_type[j])
            break

    if not stat:
        loop_type.append(nloop_types)
        nloop_types += 1


print(f"Number of loop types found: {nloop_types}")

#for i in range(nloops):
#    print(i,loop_type[i])

#visitor finds loops in order so it is assumed that the nestee appears later in the loop_types list than
#then nester.
#loop_stat = [nested(-1,-1)]*nloop_types
loop_stat = [-1]*nloop_types

for i in range(nloop_types):
#for i in range(2):
    if loop_stat[i] == -1:

        #search for presence of nested loops in body of all loops of type i
        stat = False
        for j in range(nloops):
            if loop_type[j] == i:
                for k in FindNodes(Loop).visit(loops[j].body):
                    for h in range(j+1,nloops):
                        if loops[h] == k:

                            loop_stat[i] = 0
                            l = loop_type[h]
                            
#                            print(i,h,l)

                            loop_stat[l] = 1

                            stat = True
                            break

                    if stat:
                        break

                if stat:
                    break

#print()
#for i in range(nloop_types):
#    print(i,loop_stat[i])


#loop_nest = get_nested_loops(loops[1],3)
#print("loop nest",loop_nest)




#initialize loop map
loop_map = {}
for i in range(nloops):
    loop_map.update({loops[i]:loops[i]})

#first handle isolated loops
loop_map = {}
for i in range(nloops):
    if loop_stat[loop_type[i]] == -1:
        loop_map.update({loops[i]:loops[i]})

print("nloops: ",nloops)
#now handle outer loops
for k in range(1,0,-1):
    print("k: ",k)
    loop_body = []
    for i in range(nloops):
        if loop_stat[loop_type[i]] == k:
    #       for k in FindNodes(Loop).visit(loops[i].body):
#           if loops[i] in loop_map:
           loop_body += loops[i].body
    
#    print(k,loop_body)
#    if k == 1:
#        loop_body = []
    
    j = 0
    for i in range(nloops):
        if loop_stat[loop_type[i]] == k:
            if j == 0:
                new_loop = Loop(variable=loops[i].variable, body=loop_body, bounds=loops[i].bounds)
                loop_map.update({loops[i]:new_loop})
                j += 1
            else:
                loop_map.update({loops[i]:None})


#    print(loop_map)
            
routine.body = Transformer(loop_map).visit(routine.body)

loops = FindNodes(Loop).visit(routine.body)
nloops = len(loops)
print("nloops: ",nloops)

#print(loop_map.get(loops[7]))
#print()
#print(loop_map)
#loop_map = {}
#for i in range(nloops):
##    loop_map.update({loops[i]: loops[0].clone()})
#    loop_map.update({loops[i]:comment })

#print(loop_map)

#loops = FindNodes(Loop).visit(routine.body)
print(routine.to_fortran())

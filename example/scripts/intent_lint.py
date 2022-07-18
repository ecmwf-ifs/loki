# script to lint subroutine and check if argument intent is specified correctly

from loki import Sourcefile
from loki import FindNodes,FindVariables,Loop,Assignment,CallStatement,Array,Associate,Allocation
from loki import fgen
from loki import DeferredTypeSymbol
from loki import as_tuple

import sys,os,fnmatch

def rule_check():

    #  checking if any rules are broken
    for var in in_vars:
        if not in_check[var]:
            print(f'intent(in) rule broken for {var.name}')
    
    for var in out_vars:
        if not out_check[var]:
            print(f'intent(out) rule broken for {var.name}')
    
    for var in inout_vars:
        if inout_check[var] == -1:
            print(f'intent(inout) var {var.name} can potentially be declared intent(in)')

        elif not inout_check[var]:
            print(f'intent(inout) rule broken for {var.name}')

def FindVarsNotDims(o,Return="var"):

#   retrieve all variables from expression/node
    varlist = FindVariables().visit(o)

#   build list of variables that are used as array dimensions
    dims = []
    for var in varlist:
        if isinstance(var,Array):
            dims += FindVariables().visit(var.dimensions)

#   remove duplicates from dims
    dims = list(dict.fromkeys(dims))

#   build list of var names that does not include dimensions
    if Return == "var":
        buf = [var for var in varlist if not var in dims]
    else:
        buf = [var.name for var in varlist if not var in dims]
        
    return buf
        
disable = ['TIMER','GET_ENVIRONMENT_VARIABLE','EC_PMON']

files = fnmatch.filter(os.listdir('example/src/'),'cloudsc*.F90')

routines = []
for file in files:

    print(f'read in File :: {file}')

    fnm = 'example/src/'+file

    source  = Sourcefile.from_file(fnm)
    routines += source.all_subroutines


print()
for routine in routines:

    print(f'checking {routine}\n')

    vars = FindVarsNotDims(routine.spec)
    
    # build list of variables for which intent is defined
    in_vars = []
    out_vars = []
    inout_vars = []
    
    for i,var in enumerate(vars):
        intent = getattr(var.type,"intent")
    
        if intent == "in":
           in_vars.append(var)
        elif intent == "out":
           out_vars.append(var)
        elif intent == "inout":
           inout_vars.append(var)
    
    # add associate statement to intent vars lists
    assocs = FindNodes(Associate).visit(routine.body)
    
    for assoc in assocs:
        for vars,expr in assoc.associations:
    
            varlist = FindVarsNotDims(vars)
    
            for var in varlist:

                intent = getattr(var.type,"intent")
    
                if intent == "in":
                   in_vars.append(expr)
                elif intent == "out":
                   out_vars.append(expr)
                elif intent == "inout":
                   inout_vars.append(expr)
    
    
    # check that variables with declared intent aren't used as loop induction variable
    loops = FindNodes(Loop).visit(routine.body)
    
    # intialize intent(in) rule checks: all rules must be 'True' at the end of the routine
    in_check = {var:True for var in in_vars}

    # intialize intent(out) rule checks: all rules must be 'True' at the end of the routine
    out_check = {var:True for var in out_vars}

    # intialize intent(inout) rule checks: all rules must be 'True' at the end of the routine
    inout_check = {var:True for var in inout_vars}

    for loop in loops:
        if(loop.variable in in_vars):
            in_check[loop.variable] = False
        if(loop.variable in out_vars):
            out_check[loop.variable] = False
        if(loop.variable in inout_vars):
            inout_check[loop.variable] = False
    
    print('Checking if intent variables are used as loop induction') 
    rule_check()
    
    
    
    # retrieve all the assignment nodes from the IR
    assigns = FindNodes(Assignment).visit(routine.body)
    
    # check that intent(in) vars don't appear on LHS of assignment
    for var in in_vars:
        for assign in assigns:
    
            lhs = FindVarsNotDims(assign.lhs,Return="name")

            if var.name in lhs:
                in_check[var] = False
                break
    
    
    # check that first appearance of intent(out) is not on RHS
    for var in out_vars:
        for assign in assigns:
    
            rhs = FindVarsNotDims(assign.rhs,Return="name")
            if var.name in rhs:
                out_check[var] = False
                break

            lhs = FindVarsNotDims(assign.lhs,Return="name")
            if var.name in lhs:
                break

#########.........the check that could be performed if var is genuinely intent(inout).........########    
#    inout_check = {var:2 for var in inout_vars}
#    
#    # check that intent(inout) appear first in an rhs assignment and also appear in an lhs assignment
#    for var in inout_vars:
#        for assign in assigns:
#
#            lhs = FindVarsNotDims(assign.lhs,Return="name")
#            rhs = FindVarsNotDims(assign.rhs,Return="name")
#
#            # check if var appears first only on LHS
#            if (var.name in lhs and not var.name in rhs) and inout_check[var] == 2:
#                inout_check[var] = False
#                break
#
#            if var.name in rhs and inout_check[var] == 2:
#                inout_check[var] = -1
#
#            # if var has already appeared in RHS, check it also appears on LHS
#            if var.name in lhs and inout_check[var] == -1:
#                inout_check[var] = 2
#                break
#   
#    for var in inout_vars:
#        if inout_check[var] == 2:
#            inout_check[var] = True
#        else:
#            inout_check[var] = False
    
    print('Checking intent in subroutine body') 
    rule_check()
    
    calls = FindNodes(CallStatement).visit(routine.body)
    nodes = FindNodes((CallStatement,Assignment)).visit(routine.body)

    allocs = FindNodes(Allocation).visit(routine.body)

    alloc_vars = []
    for alloc in allocs:
        alloc_vars += FindVarsNotDims(alloc.variables,Return="name")

    remove = []
    for name in disable:
        remove += [call for call in calls if name in call.name]

    calls = [call for call in calls if call not in remove]

#   associate Subroutine object with matching call
    routine.enrich_calls(routines)

#   build map of permitted values for intent(in)
    intent_map = {"in":"in"}

#   loop over all function calls
    for call in calls:

    #   retrieving subroutine object of function call
        if hasattr(call.context,'routine'):
            func = call.context.routine
        else:
            raise Exception(f'matching routine for {call} not found')
    
        for i,arg in enumerate(func.arguments):
            for var in FindVarsNotDims(arg,Return="name"):
                if var in alloc_vars:
                    
                    if getattr(arg.type,"intent") not in ['in','inout']:
                        print(f'Allocatable {var} has wrong intent in {func} declaration')
                    
    #   determine intent of all positional arguments
        intent = [None]*len(call.arguments+call.kwarguments) 
        for i,arg in enumerate(call.arguments):
    
            vars = FindVarsNotDims(arg,Return="name")
    
            for var in (in_vars+out_vars+inout_vars):
                if var.name in vars:
                    intent[i] = getattr(var.type,"intent")
    
    #   determine intent of all keyword arguments
        arglen = len(call.arguments)
        for i,(arg,value) in enumerate(call.kwarguments):
    
            vars = FindVarsNotDims(arg,Return="name")
    
            for var in (in_vars+out_vars+inout_vars):
                if var.name in vars:
                    intent[i+arglen] = getattr(var.type,"intent")
    
    #   determine location of function call relative to assignment nodes
        call_loc = [count for count,node in enumerate(nodes) if node == call][0]
    
    
    #   loop over all positional arguments and check intent consistency
        for i,arg in enumerate(call.arguments):
    
        #   determine if arg appears in assignment before function call 
            assign_type = None
            for count,node in enumerate(nodes):
                if count == call_loc: break
    
                if isinstance(node,Assignment):
    
                    lhs = FindVarsNotDims(node.lhs,Return="name")
                    rhs = FindVarsNotDims(node.rhs,Return="name")
    
                    if hasattr(arg,"name"):
                        if arg.name in lhs: assign_type = "lhs"
                        if arg.name in rhs: assign_type = "rhs"
    
            if intent[i] == "out":
    
            #   build map of permitted values for intent(out)
                if not assign_type:
                    intent_map["out"] = "out"
                elif assign_type == "lhs":
                    intent_map["out"] = ["in","inout"]    
                else:
                    intent_map["out"] = ["in","inout","out"]    
    
            elif intent[i] == "inout":
    
            #   build map of permitted values for intent(inout)
                if not assign_type:
                    intent_map["inout"] = ["in","inout","out"]
                elif assign_type == "lhs":
                    intent_map["inout"] = ["in","inout"]    
                else:
                    intent_map["inout"] = ["in","inout","out"]
    
            if intent[i] is not None:
                if getattr(func.arguments[i].type,"intent") not in intent_map[intent[i]]:   
                    print(f'Inconsistent intent in {call} for positional arg {arg.name}')
    
    #   loop over all keyword arguments and check intent consistency
        for i,(arg,value) in enumerate(call.kwarguments):
    
        #   determine if arg appears in assignment before function call 
            assign_type = None
            for count,node in enumerate(nodes):
                if count == call_loc: break
    
                if isinstance(node,Assignment):
                    lhs = FindVariables().visit(node.lhs)
                    assign_type = ["lhs" for buf in lhs if var.name == buf.name]
    
                    rhs = FindVariables().visit(node.rhs)
                    assign_type = ["rhs" for buf in rhs if var.name == buf.name]
    
                    if assign_type: assign_type = assign_type[0]
    
            if intent[i+arglen] == "out":
    
            #   build map of permitted values for intent(out)
                if not assign_type:
                    intent_map["out"] = "out"
                elif assign_type == "lhs":
                    intent_map["out"] = ["in","inout"]    
                else:
                    intent_map["out"] = ["in","inout","out"]    
    
            elif intent[i+arglen] == "inout":
    
            #   build map of permitted values for intent(inout)
                if not assign_type:
                    intent_map["inout"] = ["in","inout","out"]
                elif assign_type == "lhs":
                    intent_map["inout"] = ["in","inout"]    
                else:
                    intent_map["inout"] = ["in","inout","out"]
    
    
            #   determine position of keyword argument in subroutine declaration
            pos = [pos for pos,dum in enumerate(func.arguments) if arg == dum]
            if pos:
                pos = pos[0]
            else:
                raise Exception(f'keyword argument {arg} not found in {func} declaration')
    
            if intent[i+arglen] is not None:
                if getattr(func.arguments[pos].type,"intent") not in intent_map[intent[i]]:   
                    print(f'Inconsistent intent in {call} for keyword arg {arg.name}')

    print()

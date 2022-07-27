# script to lint subroutine and check if argument intent is specified correctly

from loki import (
  FindNodes,FindVariables,Loop,Assignment,CallStatement,Scalar,Array,Associate,Allocation,Transformer,
  Conditional,Intrinsic,SubstituteExpressions,as_tuple,convert_to_lower_case,Sourcefile,Subroutine,
  Nullify,Node,InlineCall,FindInlineCalls
 )

from loki import fgen
import sys

import os,fnmatch
import click
import pdb

def count_violations(output,summary):
    routines = []
    
    body_rule_break=['intent','rule','break']
    loop_rule_break=['intent','loop','induction']
    call_rule_break=['intent','inconsistency']
    alloc_rule_break=['intent','error','allocatable']
    
    var_unused=['intent','unused']
    inout_unused=['intent(inout)','only','intent(in)']
    io_only=['intent','I/O']
    
    unused_list=[var_unused,io_only,inout_unused]
    break_list=[body_rule_break,loop_rule_break,call_rule_break,alloc_rule_break]
    
    ucount_tot = 0
    bcount_tot = 0
    
    with open(summary,'w') as writer:
        writer.write('')
    
    with open(output,'r') as f,open(summary,'a') as writer:
        line0 = None
        for line in f:
            if 'checking Subroutine::' in line:
                routine = line.replace('checking Subroutine:: ','').strip()

                writer.write(f'checking Subroutine:: {routine}\n')

                ucount_loc = 0
                bcount_loc = 0
    
            if any(all(m in line for m in matches) for matches in break_list):
                bcount_loc += 1
                bcount_tot += 1
    
            if any(all(m in line for m in matches) for matches in unused_list):
                ucount_loc += 1
                ucount_tot += 1

            if line == '\n' and 'read in File ::' not in line0:
                writer.write(f'Intent unused:{ucount_loc}\n')
                writer.write(f'Intent violated:{bcount_loc}\n')
                writer.write('\n')

            line0 = line

    with open(summary,'a') as writer:
        writer.write('====================Total====================\n')
        writer.write(f'Intent unused:{ucount_tot}\n')
        writer.write(f'Intent violated:{bcount_tot}\n')

def rule_check(mode,var_check,output):

    for var,value in var_check.items():
        if var.type.intent in ["in","out"]:
            if not value[0]:
                print(f'intent({var.type.intent}) rule broken for {var.name}')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) rule broken for {var.name}\n')
            elif mode == "rule-unused" and value[1] == "Unused":
                print(f'intent({var.type.intent}) var {var.name} unused')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) var {var.name} unused\n')
            elif mode == "rule-unused" and value[1] == "I/O":
                print(f'intent({var.type.intent}) var {var.name} either unused or only used in I/O')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) var {var.name} either unused or only used in I/O\n')
        else:
            if not value[0]:
                print(f'intent({var.type.intent}) rule broken for {var.name}')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) rule broken for {var.name}\n')
            elif mode == "rule-unused" and value[1] == "in":
                print(f'intent({var.type.intent}) var {var.name} used only as intent(in)')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) var {var.name} used only as intent(in)\n')
            elif mode == "rule-unused" and value[1] == "Unused":
                print(f'intent({var.type.intent}) var {var.name} unused')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) var {var.name} unused\n')
            elif mode == "rule-unused" and value[1] == "I/O":
                print(f'intent({var.type.intent}) var {var.name} either unused or only used in I/O')
                with open(output,'a') as outfile:
                    outfile.write(f'intent({var.type.intent}) var {var.name} either unused or only used in I/O\n')
            
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

def FindDimsNotVars(o,Return="var"):

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
        buf = [var for var in dims]
    else:
        buf = [var.name for var in dims]
        
    return buf

def loop_check(loops,mode,ivars,var_check,output):

    for loop in loops:

        for ivar,val in ivars.items():
            if loop.variable.name in FindVarsNotDims(val,Return="name"):
                var_check[ivar][0] = False

    for ivar in ivars:
        if not var_check[ivar][0]:
            print(f'intent({ivar.type.intent}) var {ivar.name} used as loop induction')
            with open(output,'a') as outfile:
                outfile.write(f'intent({ivar.type.intent}) var {ivar.name} used as loop induction\n')


def alloc_check(calls,routine,output):

    allocs = FindNodes(Allocation).visit(routine.body)

    alloc_vars = {}
    for alloc in allocs:
        for var in FindVarsNotDims(alloc.variables):
            alloc_vars[var] = var
#            alloc_vars.update({var:var})

#   check that the allocate statement is applied directly and not on associated pointer
    for var in alloc_vars:
        if var.name not in FindVarsNotDims(routine.spec,Return="name"):
            print(f'Allocatable {var} should not be allocated via associated pointer')
            with open(output,'a') as outfile:
                outfile.write(f'Allocatable {var} should not be allocated via associated pointer\n')

    for call in calls:

        if hasattr(call.context,'routine'):
            func = call.context.routine
        else:
            raise Exception(f'matching routine for {call} not found')

        fargs = func.arguments

        for arg,v in call.kwarguments:

            for var in alloc_vars:
                if getattr(v,"name",None) in FindVarsNotDims(alloc_vars[var],Return="name"):

                    pos = [pos for pos,dum in enumerate(fargs) if arg in dum]
                    if pos:
                        pos = pos[0]
                    else:
                        raise Exception(f'keyword argument {arg} not found in {func} declaration')
              
                    farg = fargs[pos]
                    if getattr(farg.type,"intent") not in ['in','inout']:
                        print(f'intent error for allocatable {v} (kwarg) in {func} declaration')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent error for allocatable {v} (kwarg) in {func} declaration\n')

        for i,carg in enumerate(call.arguments):

            farg = fargs[i]

            for var,v in alloc_vars.items():
                if getattr(carg,"name",None) in FindVarsNotDims(v,Return="name"):
                    if getattr(farg.type,"intent") not in ['in','inout']:
                        print(f'intent error for allocatable {v} (pos arg) in {func} declaration')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent error for allocatable {v} (pos arg) in {func} declaration\n')

def call_check(calls,routine,ivars,output):

#   build map of permitted values for intent(in)
    intent_map = {"in":"in"}

    nodes = []
    for node in FindNodes((CallStatement,Assignment,Conditional,Loop)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

    for call in calls:

        if hasattr(call.context,'routine'):
            func = call.context.routine
        else:
            raise Exception(f'matching routine for {call} not found')

        args = call.arguments
        kwargs = call.kwarguments

        fargs = func.arguments

    #   determine location of function call relative to assignment nodes
        call_loc = [count for count,node in enumerate(nodes) if node == call][0]

    #   determine intent of all positional arguments
        intent = [None]*len(args+kwargs)
        for i,arg in enumerate(args):
            vars = FindVarsNotDims(arg,Return="name")

            for ivar,val in ivars.items():
                varlist = FindVarsNotDims(val,Return="name")

                for var in vars:
                    if var in varlist:
                        intent[i] = getattr(ivar.type,"intent")
    
    #   determine intent of all keyword arguments
        arglen = len(args)
        for i,(arg,value) in enumerate(kwargs):
            vars = FindVarsNotDims(value,Return="name")

            for ivar,val in ivars.items():
                varlist = FindVarsNotDims(val,Return="name")

                for var in vars:
                    if var in varlist:
                        intent[i+arglen] = getattr(ivar.type,"intent")
        
    #   loop over all positional arguments and check intent consistency
        for i,arg in enumerate(args):

        #   find ivars entry corresponding to arg
            argdict = None

            vars = FindVarsNotDims(arg,Return="name")
            for var in vars:
                for ivar,val in ivars.items():
                    varlist = FindVarsNotDims(val,Return="name")
                    if var in varlist:
                        argdict = {ivar:val[0]}
                        break
    
        #   determine if value of arg passed to function call has already been used 
            if argdict:
                assign_type = None
                for count,node in enumerate(nodes):
    
                    if count == call_loc: break
        
                    assert(len(argdict)==1)   
                    key = [*argdict][0]

                    vars = FindVarsNotDims(argdict[key],Return="name")

                    if isinstance(node,Assignment):
        
                        lhs = FindVarsNotDims(node.lhs,Return="name")
                        rhs = FindVarsNotDims(node.rhs,Return="name")
 
                        for l in lhs:
                            if l in vars: assign_type = "lhs"
                        for r in rhs:
                            if r in vars: assign_type = "rhs"

                    elif isinstance(node,CallStatement):

                        for a in node.arguments:
                            rhs = [buf.name for buf in FindVariables().visit(a)]
                            for r in rhs:
                                if r in vars: assign_type = "rhs"

                        for a,v in node.kwarguments:
                            rhs = [buf.name for buf in FindVariables().visit(v)]
                            for r in rhs:
                                if r in vars: assign_type = "rhs"

                    elif isinstance(node,Loop):

                        bounds = FindVarsNotDims(node.bounds)

                        for b in bounds:
                            rhs = [buf.name for buf in FindVariables().visit(b)]
                            for r in rhs:
                                if r in vars: assign_type = "rhs"

                    elif isinstance(node,Conditional):
 
                        rhs = [buf.name for buf in FindVariables().visit(node.condition)]
                        for r in rhs:
                            if r in vars: assign_type = "rhs"

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
                        
                    if getattr(fargs[i].type,"intent") not in intent_map[intent[i]]:   
                        print(f'intent inconsistency in {call} for positional arg {arg.name}')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent inconsistency in {call} for positional arg {arg.name}\n')
    
    #   loop over all keyword arguments and check intent consistency
        for i,(arg,value) in enumerate(kwargs):

        #   find ivars entry corresponding to arg
            argdict = None

            vars = FindVarsNotDims(value,Return="name")
            for var in vars:
                for ivar,val in ivars.items():
                    varlist = FindVarsNotDims(val,Return="name")
                    if var in varlist:
                        argdict = {ivar:val[0]}
                        break
    
        #   determine if value of arg passed to function call has already been used 
            if argdict:
                assign_type = None
                for count,node in enumerate(nodes):

                    if count == call_loc: break
            
                    if isinstance(node,Assignment):
                        lhs = FindVarsNotDims(node.lhs,Return="name")
                        rhs = FindVarsNotDims(node.rhs,Return="name")
 
                        assert(len(argdict)==1)   
                        key = [*argdict][0]

                        if hasattr(key,"name"):
                            vars = FindVarsNotDims(argdict[key],Return="name")

                            for l in lhs:
                                if l in vars: assign_type = "lhs"
                            for r in rhs:
                                if r in vars: assign_type = "rhs"

        
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
                pos = [pos for pos,dum in enumerate(fargs) if arg in dum]
                if pos:
                    pos = pos[0]
                else:
                    raise Exception(f'keyword argument {arg} not found in {func} declaration')
        
                if intent[i+arglen] is not None:

                    if getattr(fargs[pos].type,"intent") not in intent_map[intent[i+arglen]]:
                        print(f'intent inconsistency in {call} for keyword arg {arg}')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent inconsistency in {call} for keyword arg {arg}\n')

def body_check(routine,mode,in_vars,out_vars,inout_vars,var_check):

    # check if intent(in) or intent(inout) vars are used in loop bounds
    loops = FindNodes(Loop).visit(routine.body)
    for ivar,value in ({**in_vars,**inout_vars}).items():

        for loop in loops:
    
            bounds = FindVarsNotDims(loop.bounds)

            for bound in bounds:
    
                if bound.name in FindVarsNotDims(value,Return="name"):
                    if ivar.type.intent == "in":
                        var_check[ivar][1] = "Used"
                    elif var_check[ivar][1] != "Used":
                        var_check[ivar][1] = "in"
     
            if not var_check[ivar][1] == "Unused":
                break

    # check intent(in) vars: don't appear on LHS of assignment, used in conditional
    nodes = []
    for node in FindNodes((Assignment,Conditional)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

    for key,value in in_vars.items():
        vars = FindVarsNotDims(value,Return="name")
        for node in nodes:
   
            if isinstance(node,Assignment):
                lhs = FindVarsNotDims(node.lhs,Return="name")
                for l in lhs:
                    if l in vars:
                        var_check[key][0] = False
                        break
    
                rhs = FindVarsNotDims(node.rhs,Return="name")
                for r in rhs:
                    if r in vars:
                        var_check[key][1] = "Used"

                dims = FindDimsNotVars(node.lhs,Return="name")
                dims += FindDimsNotVars(node.rhs,Return="name")
                for d in dims:
                    if d in vars:
                        var_check[key][1] = "Used"

            else:
                cond = FindVarsNotDims(node.condition,Return="name")
                for c in cond:
                    if c in vars:
                        var_check[key][1] = "Used"

#    pdb.set_trace()
    # check if intent(in,inout) vars are used in callstatements
    for call in FindNodes(CallStatement).visit(routine.body):

        if hasattr(call.context,'routine'):
            func = call.context.routine
            fargs = func.arguments
    
            for i,arg in enumerate(call.arguments):
                vars = FindVarsNotDims(arg,Return="name")
                dims = FindDimsNotVars(arg,Return="name")
    
                for k,v in in_vars.items():
                    varlist = FindVarsNotDims(v,Return="name")
                    for var in vars+dims:
                        if var in varlist:
                            var_check[k][1] = "Used"
    
                for k,v in inout_vars.items():
                    varlist = FindVarsNotDims(v,Return="name")
    
                    for var in vars:
                        if var in varlist:
                            if getattr(fargs[i].type,"intent") == "in" and var_check[k][1] != "Used":
                                var_check[k][1] = "in"
                            else:
                                var_check[k][1] = "Used"
    
                    for dim in dims:
                        if dim in varlist and var_check[k][1] != "Used":
                            var_check[k][1] = "in"
    
            for i,(arg,value) in enumerate(call.kwarguments):
                vars = FindVarsNotDims(value,Return="name")
                dims = FindDimsNotVars(value,Return="name")
    
                #   determine position of keyword argument in subroutine declaration
                pos = [pos for pos,dum in enumerate(fargs) if arg in dum]
                if pos:
                    pos = pos[0]
                else:
                    raise Exception(f'keyword argument {arg} not found in {func} declaration')
    
                for k,v in in_vars.items():
                    varlist = FindVarsNotDims(v,Return="name")
                    for var in vars+dims:
                        if var in varlist:
                            var_check[k][1] = "Used"
    
                for k,v in inout_vars.items():
                    varlist = FindVarsNotDims(v,Return="name")
    
                    for var in vars:
                        if var in varlist:
                            if getattr(fargs[pos].type,"intent") == "in" and var_check[k][1] != "Used":
                                var_check[k][1] = "in"
                            else:
                                var_check[k][1] = "Used"
    
                    for dim in dims:
                        if dim in varlist and var_check[k][1] != "Used":
                            var_check[k][1] = "in"
        else:
            for i,arg in enumerate(call.arguments):
                vars = FindVariables().visit(arg)
    
                for k,v in {**in_vars,**inout_vars}.items():
                    varlist = FindVarsNotDims(v,Return="name")
                    for var in vars:
                        if var in varlist:
                            var_check[k][1] = "Used"
            for i,(arg,value) in enumerate(call.kwarguments):
                vars = FindVariables().visit(value)
    
                for k,v in {**in_vars,**inout_vars}.items():
                    varlist = FindVarsNotDims(v,Return="name")
                    for var in vars:
                        if var in varlist:
                            var_check[k][1] = "Used"

    # check if intent(inout) var ever appears on LHS
    nodes = []
    for node in FindNodes((Assignment,Conditional)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

    for key,value in inout_vars.items():
        vars = FindVarsNotDims(value,Return="name")
        for node in nodes:

            if isinstance(node,Assignment):
                rhs = FindVarsNotDims(node.rhs,Return="name")

                for r in rhs:
                    if r in vars and var_check[key][1] != "Used":
                        var_check[key][1] = "in"
    
                dims = FindDimsNotVars(node.rhs,Return="name")
                dims += FindDimsNotVars(node.lhs,Return="name")
                for d in dims:
                    if d in vars and var_check[key][1] != "Used":
                        var_check[key][1] = "in"

                lhs = FindVarsNotDims(node.lhs,Return="name")
                for l in lhs:
                    if l in vars:
                        var_check[key][1] = "Used"
    
            else:
                cond = FindVarsNotDims(node.condition,Return="name")
                for c in cond:
                    if c in vars and var_check[key][1] != "Used":
                        var_check[key][1] = "in"

            if var_check[key][1] == "Used": break

    # check rules for intent(out) vars
    nodes = []
    for node in FindNodes((Assignment,Conditional,CallStatement,Loop)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

#    pdb.set_trace()
    for key,value in out_vars.items():
        vars = FindVarsNotDims(value,Return="name")
        for node in nodes:

            if isinstance(node,CallStatement):
                
                for i,arg in enumerate(node.arguments):
                
                    args = FindVarsNotDims(arg,Return="name")
        
                    for a in args:
                        if a in vars:
                            var_check[key][1] = "Used"

                for i,(arg,value) in enumerate(node.kwarguments):
                
                    args = FindVarsNotDims(value,Return="name")
        
                    for a in args:
                        if a in vars:
                            var_check[key][1] = "Used"

            elif isinstance(node,Assignment):

                rhs = FindVarsNotDims(node.rhs,Return="name")

                for r in rhs:
                    if r in vars:
                        var_check[key][0] = False

                lhs = FindVarsNotDims(node.lhs,Return="name")

                for l in lhs:
                    if l in vars:
                        var_check[key][1] = "Used"
    
            elif isinstance(node,Conditional):
                icalls = FindInlineCalls().visit(node.condition)
                exclude = ()

                for icall in icalls:
                    if icall.name.lower() in ['ubound','size','present']:

                        cond = [var.lower() for var in FindVarsNotDims(icall,Return="name")]

                        for c in cond:
                            if c in vars:
                                var_check[key][1] = "InlineCall"
                        exclude += as_tuple(FindVariables().visit(icall))

                if len(exclude)>0:
                    condition = as_tuple([var for var in FindVariables().visit(node.condition) if not var in exclude])

                if var_check[key][1] in ["Unused","InlineCall"]:
                    if len(exclude)>0:
                        cond = []
                        for c in condition:
                            cond += [var.lower() for var in FindVarsNotDims(c,Return="name")]
                    else:
                        cond = [var.lower() for var in FindVarsNotDims(node.condition,Return="name")]

                    for c in cond:
                        if c in vars:
                            var_check[key][0] = False

            elif isinstance(node,Loop):
                bounds = FindVarsNotDims(node.bounds)
    
                for bound in bounds:
                    if bound.name in vars: 
                        var_check[key][0] = False

            if not var_check[key][0] or var_check[key][1] == "Used":
                break
 
            if var_check[key][1] == "InlineCall": var_check[key][1] == "Unused"

def spec_check(routine,vars,var_check):

#   check if var used as dimension in routine spec 
    dims = FindDimsNotVars(routine.spec)

    for dim in dims:
        if dim in vars:
            if dim.type.intent == "in":
                var_check[dim][1] = "Used"
            elif var_check[dim] != "Used": 
                var_check[dim][1] = "in"

#   check if var used as dimension in allocation
    allocs = FindNodes(Allocation).visit(routine.body)

    for alloc in allocs:
        for dim in FindDimsNotVars(alloc.variables):
            for var,val in vars.items():
                if dim in val: 
                    if dim.type.intent == "in":
                        var_check[dim][1] = "Used"
                    elif var_check[dim] != "Used": 
                        var_check[dim][1] = "in"

def resolve_member_routine(routine,var_check,in_vars,out_vars,inout_vars,disable):

    mem_routines = as_tuple([r for r in routine.contains.body if isinstance(r,Subroutine)])
    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name.name.lower() in str(mem_routines).lower()]

    rglob = as_tuple([call.context.routine.clone() for call in calls])
    call_map = {}
    for count,call in enumerate(calls):

#        if call.kwarguments:
#            f'double-check kwargs in {call}  to internal routine in {routine}'

        cargs = call.arguments

        r = rglob[count]
        convert_to_lower_case(r)

        for arg in r.arguments:
            for a,v in call.kwarguments:
                if arg.name.lower() == a.lower():
                    cargs += as_tuple(v)

        # rename dummy arguments in r.body
        vmap = {}
        tmp_vars=()

        spec_vars = [var.name for var in FindVariables().visit(r.spec)]
        vars = [var for var in FindVariables().visit(r.body) if var.name != "zhook_handle" and var.name in spec_vars]
        for var in vars:

            stat = False
            for j,arg in enumerate(r.arguments):
                if arg.name in var.name:
                    vmap[var] = var.clone(name=cargs[j].name)
                    stat = True

            if not stat:
                tmp_vars += as_tuple(var.clone(name=(var.name+f'_{r.name}_tmp_{count}')))
                vmap[var] = tmp_vars[-1]

        r.body = SubstituteExpressions(vmap).visit(r.body)

        r.rescope_symbols()
        r.variables = tmp_vars

        # remove disabled callstatements
        rcalls = [rcall for rcall in FindNodes(CallStatement).visit(r.body) if rcall.name.name in disable]
        cmap = {}
        for rcall in rcalls:
            cmap[rcall] = None
        r.body = Transformer(cmap).visit(r.body)

        # remove calls to LHOOK
        conds = [cond for cond in FindNodes(Conditional).visit(r.body) if 'LHOOK' in str(cond.condition)]
        cmap = {}
        for cond in conds:
            cmap[cond] = None
        r.body = Transformer(cmap).visit(r.body)

        call_map[call] = r.body

        routine.variables = routine.variables + tmp_vars
        routine.rescope_symbols()

        # add new vars to intent var list
        for var in tmp_vars:
            if getattr(var.type,"intent") == "in":
                in_vars[var] = var
            elif getattr(var.type,"intent") == "out":
                out_vars[var] = var
            elif getattr(var.type,"intent") == "inout":
                inout_vars[var] = var

            if getattr(var.type,"intent"):
                var_check[var] = [True,"Unused"]


    routine.body = Transformer(call_map).visit(routine.body)

def resolve_association(assoc_map,assoc,all_vars,ivars):
    vmap = {}
    for rexpr,l in assoc.associations:
    
        vars = FindVariables().visit(rexpr)
    
        lexpr = ()
        for var in all_vars:
            if isinstance(var,Array) and var.name == l.name.lower():
                lexpr += as_tuple(var)
    
        lexpr += as_tuple(l)
    
        for var in vars:
            for ivar,val in ivars.items():
    
                if var.name in FindVarsNotDims(val,Return="name"):
    
                    for l in lexpr:
                        if isinstance(l,Array):
                            r=Array(name=rexpr.name.lower(),dimensions=l.dimensions,type=l.scope.symbol_attrs[l.name],scope=l.scope)
                            vmap[l] = r
                        else: 
                            vmap[l] = rexpr
    
        assoc.body = SubstituteExpressions(vmap).visit(assoc.body)

def resolve_pointer(assign,routine,ivars,onodes,nnodes,nmap):

    start = None
    start = [node for node in FindNodes(Assignment).visit(routine.body) if node == assign]

    assert start, f'Pointer {assign} not found'
    if start:
        start = start[0]

    vars = FindVarsNotDims(assign.lhs,Return="name")

    end = None
    stat = False
    for node in FindNodes((Nullify,Assignment)).visit(routine.body):

        if node == start:
           stat = True
        if stat:
            if isinstance(node,Nullify):
                if any(x in vars for x in FindVarsNotDims(node.variables,Return="name")):
                    end = node
                    break
            elif getattr(node,"ptr",None):
                if any(x in vars for x in FindVarsNotDims(node.lhs,Return="name")):
                    if 'null' in [var.lower() for var in FindVarsNotDims(node.rhs,Return="name")]:
                       end = node
                       break


    ist = None
    ien = None
    for count,node in enumerate(nnodes):
        if node == start:
            ist = count
        if node == end:
            ien = count
    
    if not ien:
        ien = len(nnodes)

    vmap = {}
    for count,node in enumerate(nnodes):
        if count >= ist and count < ien:
            
            vars = FindVariables().visit(assign.rhs)

            for ivar,val in ivars.items():
                for var in [var for var in vars if var.name in FindVarsNotDims(val,Return="name")]:
            
                    for v in FindVariables().visit(node):
                        if v.name in [l.name for l in as_tuple(assign.lhs)]:
                            vmap[v] = assign.rhs
        
            node = SubstituteExpressions(vmap).visit(node)
            nmap[onodes[count]] = node

@click.command(help='Check if dummy argument INTENT is consistent with how variables are used')
@click.option('--mode',default='rule-break',type=click.Choice(['rule-break','rule-unused'],case_sensitive=False),help=('rule-break: check only for intent violations. rule-unused: also check for redundant intent'))
@click.option('--intype',type=click.Choice(['path','file'],case_sensitive=False),help=("path: specify file path directly. file: read in file paths from input file"))
@click.option('--path',type=str,help=('Path of file(s) to be parsed or path of input file'))
@click.option('--disable','-d',type=str,multiple=True,default=None,help=('List of function calls to be excluded from intent check'))
@click.option('--output',type=str,default='output.dat',help=('Path of file to write raw output'))
@click.option('--summary',type=str,default='summary.dat',help=('Path of file to write counted violations'))
def main(mode,intype,path,disable,output,summary):

    files = []
    if intype == 'path':
        if '/' in path:
            files = fnmatch.filter(os.listdir('/'.join(path.split('/')[:-1])+'/'),path.split('/')[-1])
            files = ['/'.join(path.split('/')[:-1])+'/'+file for file in files]
        else:
            files = fnmatch.filter(os.listdir('.'),path)
    else:
        with open(path,'r') as reader:
            for line in reader:
                if not line[0] == '#':
                    files.append(line.strip())

        exclude = []
        with open(disable[0],'r') as reader:
            for line in reader:
                if not line[0] == '#':
                    exclude.append(line.strip())

        disable = exclude

    routines = []
    sources = len(files)*[None]

    with open(output,'w') as outfile:
        outfile.write('')

    for n,file in enumerate(files):
    
        print(f'read in File :: {file}')

        with open(output,'a') as outfile:
            outfile.write(f'read in File :: {file}\n')
     
        sources[n] = Sourcefile.from_file(file)
        routines += sources[n].all_subroutines
 
    print()
    with open(output,'a') as outfile:
        outfile.write('\n')

    for routine in routines:
    
        print(f'checking {routine}')
        with open(output,'a') as outfile:
            outfile.write(f'checking {routine}\n')

        # convert entire routine to lowercase
        convert_to_lower_case(routine)

        # build dicts of variables for which intent is defined
        in_vars = {}
        out_vars = {}
        inout_vars = {}
        
        vars = FindVariables().visit(routine.spec)

        for i,var in enumerate(vars):
            intent = getattr(var.type,"intent")
        
            if intent == "in":
               in_vars.update({var:var})
            elif intent == "out":
               out_vars.update({var:var})
            elif intent == "inout":
               inout_vars.update({var:var})

        ivars = {**in_vars,**out_vars,**inout_vars}


        # resolving associations
        assoc_map = {} 
        all_vars = FindVariables().visit(routine.body)
        onodes = FindNodes(Node).visit(routine.body) #old nodes
        nnodes = FindNodes(Node).visit(routine.body) #new nodes
        for pointer in FindNodes((Associate,Assignment)).visit(routine.body):

            if isinstance(pointer,Assignment):
                if getattr(pointer,"ptr",None):
                    resolve_pointer(pointer,routine,ivars,onodes,nnodes,assoc_map)
            else:
                resolve_association(assoc_map,pointer,all_vars,ivars)
                assoc_map[pointer] = pointer.body

        routine.body = Transformer(assoc_map).visit(routine.body)
        routine.rescope_symbols()

#        print(fgen(routine.body))

        # intialize intent rule checks
        var_check = {var:[True,"Unused"] for var in {**in_vars,**out_vars,**inout_vars}}

        #  associate Subroutine object with matching call
        routine.enrich_calls(routines)

        if routine.contains:
            for node in routine.contains.body:
                if isinstance(node,Subroutine):
                    routine.enrich_calls(node)


        # remove specified functions from call list
        calls = FindNodes(CallStatement).visit(routine.body)
    
        remove = []
        for name in disable:
            remove += [call for call in calls if name.lower() in call.name.name.lower()]

        calls = [call for call in calls if call not in remove]

        #  checking the intent of allocatable vars passed as dummy arguments
        alloc_check(calls,routine,output)
    
        #  checking the intent consistency across function calls
        call_check(calls,routine,{**in_vars,**out_vars,**inout_vars},output)

####........resolve internal subroutines here........####
        if routine.contains:
            resolve_member_routine(routine,var_check,in_vars,out_vars,inout_vars,disable)

        # check that variables with declared intent aren't used as loop induction variable
        loops = FindNodes(Loop).visit(routine.body)
        loop_check(loops,mode,{**in_vars,**out_vars,**inout_vars},var_check,output)

        # check if intent(in,inout) variables are used to define array size
        spec_check(routine,{**in_vars,**inout_vars},var_check)

        #  checking intent rules in subroutine body
        body_check(routine,mode,in_vars,out_vars,inout_vars,var_check)

        #  checking for I/O
        intrinsics = FindNodes(Intrinsic).visit(routine.body)
        for var in {**in_vars,**out_vars,**inout_vars}:
            if var_check[var][1] != "Used":
                for intr in intrinsics:
                    if var.name in str.lower(intr.text):
                        var_check[var][1] = "I/O"

        rule_check(mode,var_check,output)
        print()
        with open(output,'a') as outfile:
            outfile.write('\n')

    count_violations(output,summary)

if __name__ == "__main__":
   main()

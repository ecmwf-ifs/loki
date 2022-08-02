# script to lint subroutine and check if argument intent is specified correctly

from loki import Sourcefile
from loki import (
   FindNodes,FindVariables,Loop,Assignment,CallStatement,Array,Associate,Allocation,Transformer,
   Conditional,Intrinsic)
from loki import SubstituteExpressions
from loki import convert_to_lower_case

import os,fnmatch
import click

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
#        writer.write('\n')
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
            alloc_vars.update({var:var})

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

        for arg,v in call.kwarguments:

            for var in alloc_vars:
                if getattr(v,"name",None) in FindVarsNotDims(alloc_vars[var],Return="name"):

                    pos = [pos for pos,dum in enumerate(func.arguments) if arg in dum]
                    if pos:
                        pos = pos[0]
                    else:
                        raise Exception(f'keyword argument {arg} not found in {func} declaration')
              
                    farg = func.arguments[pos]
                    if getattr(farg.type,"intent") not in ['in','inout']:
                        print(f'intent error for allocatable {v} (kwarg) in {func} declaration')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent error for allocatable {v} (kwarg) in {func} declaration\n')

        for i in range(len(call.arguments)):

            farg = func.arguments[i]
            carg = call.arguments[i]

            for var,v in alloc_vars.items():
                if getattr(carg,"name",None) in FindVarsNotDims(v,Return="name"):
                    if getattr(farg.type,"intent") not in ['in','inout']:
                        print(f'intent error for allocatable {v} (pos arg) in {func} declaration')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent error for allocatable {v} (pos arg) in {func} declaration\n')

def call_check(calls,routine,ivars,var_check,output):

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

    #   determine intent of all positional arguments
        intent = [None]*len(call.arguments+call.kwarguments)

    #   determine intent of all keyword arguments
        arglen = len(call.arguments)

    #   determine location of function call relative to assignment nodes
        call_loc = [count for count,node in enumerate(nodes) if node == call][0]

        for i,arg in enumerate(call.arguments):
        
            vars = FindVarsNotDims(arg,Return="name")
            dims = FindDimsNotVars(arg,Return="name")

            for ivar,val in ivars.items():
                varlist = FindVarsNotDims(val,Return="name")

                for var in vars:
                    if var in varlist:
                        var_check[ivar][1] = "Used"
                        intent[i] = getattr(ivar.type,"intent")

                for dim in dims:
                    if dim in varlist:
                        if ivar.type.intent in ["in","out"]:
                            var_check[ivar][1] = "Used"
                        else:
                            var_check[ivar][1] = "in"
    
        for i,(arg,value) in enumerate(call.kwarguments):
        
            vars = FindVarsNotDims(value,Return="name")
            dims = FindDimsNotVars(value,Return="name")

            for ivar,val in ivars.items():
                varlist = FindVarsNotDims(val,Return="name")

                for var in vars:
                    if var in varlist:
                        var_check[ivar][1] = "Used"
                        intent[i+arglen] = getattr(ivar.type,"intent")
        
                for dim in dims:
                    if dim in varlist:
                        if ivar.type.intent in ["in","out"]:
                            var_check[ivar][1] = "Used"
                        else:
                            var_check[ivar][1] = "in"
    
    #   loop over all positional arguments and check intent consistency
        for i,arg in enumerate(call.arguments):

        #   find ivars entry corresponding to arg
            argdict = None

            vars = FindVarsNotDims(arg,Return="name")
            for var in vars:
                for ivar,val in ivars.items():
                    varlist = FindVarsNotDims(val,Return="name")
                    if var in varlist:
                        argdict = {ivar:val[0]}
                        break
    
        #   determine if arg appears in assignment before function call 
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
                    if intent[i] == "inout" and getattr(func.arguments[i].type,"intent") == "in":
                        for ivar in ivars:
                            if ivar.name == func.arguments[i].name:
                                var_check[ivar][1] = "in"
                        
                    if getattr(func.arguments[i].type,"intent") not in intent_map[intent[i]]:   
                        print(f'intent inconsistency in {call} for positional arg {arg.name}')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent inconsistency in {call} for positional arg {arg.name}\n')
    
    #   loop over all keyword arguments and check intent consistency
        for i,(arg,value) in enumerate(call.kwarguments):

        #   find ivars entry corresponding to arg
            argdict = None

            vars = FindVarsNotDims(value,Return="name")
            for var in vars:
                for ivar,val in ivars.items():
                    varlist = FindVarsNotDims(val,Return="name")
                    if var in varlist:
                        argdict = {ivar:val[0]}
                        break
    
        #   determine if arg appears in assignment before function call
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
                pos = [pos for pos,dum in enumerate(func.arguments) if arg in dum]
                if pos:
                    pos = pos[0]
                else:
                    raise Exception(f'keyword argument {arg} not found in {func} declaration')
        
                if intent[i+arglen] is not None:
                    if intent[i] == "inout" and getattr(func.arguments[pos].type,"intent") == "in":
                        for ivar in ivars:
                            if ivar.name == func.arguments[pos].name:
                                var_check[ivar][1] = "in"

                    if getattr(func.arguments[pos].type,"intent") not in intent_map[intent[i]]:
                        print(f'intent inconsistency in {call} for keyword arg {arg}')
                        with open(output,'a') as outfile:
                            outfile.write(f'intent inconsistency in {call} for keyword arg {arg}\n')

def body_check(routine,calls,mode,in_vars,out_vars,inout_vars,var_check):

    nodes = []
    for node in FindNodes((Assignment,Conditional)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

    loops = FindNodes(Loop).visit(routine.body)


    # check that intent(in) vars don't appear on LHS of assignment
    for key,value in in_vars.items():
        for node in nodes:
            vars = FindVarsNotDims(value,Return="name")
   
            if isinstance(node,Assignment):
                lhs = FindVarsNotDims(node.lhs,Return="name")
                for l in lhs:
                    if l in vars:
                        var_check[key][0] = False
    
                if not var_check[key][0]: break

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

    # check if intent(in) or intent(inout) vars are used in loop bounds
    for ivar,value in ({**in_vars,**inout_vars}).items():

        if var_check[ivar][1] == "Unused":
            for loop in loops:
    
                bounds = FindVarsNotDims(loop.bounds)

                for bound in bounds:
    
                    if bound.name in FindVarsNotDims(value,Return="name"):
                        if ivar.type.intent == "in":
                            var_check[ivar][1] = "Used"
                        else:
                            var_check[ivar][1] = "in"
         
                if not var_check[ivar][1] == "Unused":
                    break
     

    # reinitialize "Used" status for out_vars
    for var in out_vars:
        var_check[var][1] = "Unused"

    nodes = []
    for node in FindNodes((Assignment,Conditional,CallStatement,Loop)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

    # check rules for intent(out) vars
    for key,value in out_vars.items():
        for node in nodes:
            vars = FindVarsNotDims(value,Return="name")

            if isinstance(node,CallStatement) and var_check[key][1] == "Unused":
                
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

                if var_check[key][1] == "Unused":
                    rhs = FindVarsNotDims(node.rhs,Return="name")

                    for r in rhs:
                        if r in vars:
                            var_check[key][0] = False

                lhs = FindVarsNotDims(node.lhs,Return="name")

                for l in lhs:
                    if l in vars:
                        var_check[key][1] = "Used"
    
            elif isinstance(node,Conditional):
                cond = FindVarsNotDims(node.condition,Return="name")
                for c in cond:
                    if c in vars:
                        var_check[key][1] = "Used"

            elif isinstance(node,Loop):
                bounds = FindVarsNotDims(node.bounds)
    
                for bound in bounds:
                    if bound.name in vars: 
                        var_check[key][1] = "Used"

            if not var_check[key][0] or var_check[key][1] == "Used":
                break
 
    nodes = []
    for node in FindNodes((Assignment,Conditional)).visit(routine.body):
        if isinstance(node,Assignment):
            if not node.ptr:
                nodes.append(node)
        else:
            nodes.append(node)

    # check if intent(intout) var ever appears on LHS
    for key,value in inout_vars.items():
        if var_check[key][1] == "Unused":
            for node in nodes:
                vars = FindVarsNotDims(value,Return="name")
    
                if isinstance(node,Conditional):
                    cond = FindVarsNotDims(node.condition,Return="name")
                    for c in cond:
                        if c in vars:
                            var_check[key][1] = "in"
                else:

                    rhs = FindVarsNotDims(node.rhs,Return="name")

                    for r in rhs:
                        if r in vars:
                            var_check[key][1] = "in"
    
                    dims = FindDimsNotVars(node.rhs,Return="name")
                    dims += FindDimsNotVars(node.lhs,Return="name")
                    for d in dims:
                        if d in vars:
                            var_check[key][1] = "in"
    
        for node in nodes:
            if isinstance(node,Assignment):
                lhs = FindVarsNotDims(node.lhs,Return="name")
                for l in lhs:
                    if l in vars:
                        var_check[key][1] = "Used"
                        break

def spec_check(routine,vars,var_check):

#   check if var used as dimension in routine spec 
    dims = FindDimsNotVars(routine.spec)

    for dim in dims:
        if dim in vars:
            if dim.type.intent == "in":
                var_check[dim][1] = "Used"
            else: 
                var_check[dim][1] = "in"

#   check if var used as dimension in allocation
    allocs = FindNodes(Allocation).visit(routine.body)

    for alloc in allocs:
        for dim in FindDimsNotVars(alloc.variables):
            for var,val in vars.items():
                if dim in val: 
                    if dim.type.intent == "in":
                        var_check[dim][1] = "Used"
                    else: 
                        var_check[dim][1] = "in"

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
                    exclude.append(str.lower(line.strip()))

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


        # resolving associations declared through the associate clause
        assocs = FindNodes(Associate).visit(routine.body)
        assoc_map = {}
        vmap = {}

        for assoc in assocs:
            for rexpr,lexpr in assoc.associations:
    
                vars = FindVarsNotDims(rexpr)
    
                for var in vars:
                    for ivar,val in ivars.items():
 
                        if var.name in FindVarsNotDims(val,Return="name"):
                            vmap[lexpr] = rexpr

                assoc.body = SubstituteExpressions(vmap).visit(assoc.body)
            assoc_map[assoc] = assoc.body
        routine.body = Transformer(assoc_map).visit(routine.body)

        # resolving associations declared through pointer association
        assigns = FindNodes(Assignment).visit(routine.body)
        assigns = [assign for assign in assigns if getattr(assign,"ptr",None)] 
        for assign in assigns:
            if getattr(assign,"ptr",None):
    
                vars = FindVarsNotDims(assign.rhs)
        
                for var in vars:
                    for ivar,val in ivars.items():
        
                        if var.name in FindVarsNotDims(val,Return="name"):
                            vmap[assign.lhs] = assign.rhs
                routine.body = SubstituteExpressions(vmap).visit(routine.body)

        routine.rescope_symbols()

        # intialize intent rule checks
        var_check = {var:[True,"Unused"] for var in {**in_vars,**out_vars,**inout_vars}}

        #  associate Subroutine object with matching call
        routine.enrich_calls(routines)

        #  get all associate blocks
        assocs = FindNodes(Associate).visit(routine.body)

        # check that variables with declared intent aren't used as loop induction variable
        loops = FindNodes(Loop).visit(routine.body)
        loop_check(loops,mode,{**in_vars,**out_vars,**inout_vars},var_check,output)

        # check if intent(in,inout) variables are used to define array size
        spec_check(routine,{**in_vars,**inout_vars},var_check)

        # remove specified functions from call list
        calls = FindNodes(CallStatement).visit(routine.body)
    
        remove = []
        for name in disable:
            remove += [call for call in calls if name in str.lower(call.name.name)]

        calls = [call for call in calls if call not in remove]

        #  checking the intent of allocatable vars passed as dummy arguments
        alloc_check(calls,routine,output)
    
        #  checking the intent consistency across function calls
        call_check(calls,routine,{**in_vars,**out_vars,**inout_vars},var_check,output)
    
        #  checking intent rules in subroutine body
        body_check(routine,calls,mode,in_vars,out_vars,inout_vars,var_check)

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

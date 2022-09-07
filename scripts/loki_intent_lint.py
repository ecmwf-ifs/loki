# script to lint subroutine and check if argument intent is specified correctly

import os
import sys
import pdb
import toml
import click
import fnmatch

from pathlib import Path, PurePath
from collections import defaultdict

from loki import (
  FindNodes, FindVariables, Loop, Assignment, CallStatement, Scalar, Array, Associate, Allocation,
  Transformer, Conditional, Intrinsic, SubstituteExpressions, as_tuple, convert_to_lower_case, 
  Sourcefile, Subroutine, Nullify, Node, InlineCall, FindInlineCalls, flatten, Section, Scheduler,
  LeafNode, InternalNode, fgen, dataflow_analysis_attached
  )


def count_violations(output, summary):
    routines = []
    
    body_rule_break=['intent', 'rule', 'broken']
    loop_rule_break=['intent', 'loop', 'induction']
    call_rule_break=['intent', 'inconsistency']
    alloc_rule_break=['Allocatable', 'wrong', 'intent']
    spec_rule_break=['intent(out)', 'subroutine', 'specification']
    
    var_unused=['intent', 'unused']
    inout_unused=['intent(inout)', 'only', 'intent(in)']
    io_only=['intent', 'I/O']

    intent_undeclared=['Dummy', 'no declared', 'intent']
    
    unused_list=[var_unused, io_only, inout_unused]
    break_list=[body_rule_break, loop_rule_break, call_rule_break, alloc_rule_break, spec_rule_break]
    undeclared_list=[intent_undeclared]
    
    ucount_tot = 0
    bcount_tot = 0
    dcount_tot = 0
    
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
                dcount_loc = 0
    
            if any(all(m in line for m in matches) for matches in break_list):
                bcount_loc += 1
                bcount_tot += 1
    
            if any(all(m in line for m in matches) for matches in unused_list):
                ucount_loc += 1
                ucount_tot += 1

            if any(all(m in line for m in matches) for matches in undeclared_list):
                dcount_loc += 1
                dcount_tot += 1

            if line == '\n' and not any(s in line0 for s in ('read in File ::','collected from scheduler')):
                writer.write(f'Intent undeclared:{dcount_loc}\n')
                writer.write(f'Intent unused:{ucount_loc}\n')
                writer.write(f'Intent violated:{bcount_loc}\n')
                writer.write('\n')

            line0 = line

    with open(summary, 'a') as writer:
        writer.write('====================Total====================\n')
        writer.write(f'Intent undeclared:{dcount_tot}\n')
        writer.write(f'Intent unused:{ucount_tot}\n')
        writer.write(f'Intent violated:{bcount_tot}\n')

def rule_check(mode, var_check, output, routine):
    """Checks which vars have either broken intent rules or have unused intent."""

    for var, value in var_check.items():
        intent = routine.symbol_map[var].type.intent
        if not value[0]:
            print(f'intent({intent}) rule broken for {var}')
            if output:
                with open(output, 'a') as outfile:
                    outfile.write(f'intent({intent}) rule broken for {var}\n')
        elif mode == "rule-unused" and value[1] == "Unused":
            print(f'intent({intent}) var {var} unused')
            if output:
                with open(output, 'a') as outfile:
                    outfile.write(f'intent({intent}) var {var} unused\n')
        elif mode == "rule-unused" and value[1] == "in":
            print(f'intent({intent}) var {var} used only as intent(in)')
            if output:
                with open(output, 'a') as outfile:
                    outfile.write(f'intent({intent}) var {var} used only as intent(in)\n')
        elif mode == "rule-unused" and value[1] == "I/O":
            print(f'intent({intent}) var {var} either unused or only used in I/O')
            if output:
                with open(output, 'a') as outfile:
                    outfile.write(f'intent({intent}) var {var} either unused or only used in I/O\n')

def findvarsnotdims(o, return_vars=True):
    """
    Returns only the arrays and scalars present in an expression or node, i.e. excluding dimensions.
    """

    dims = flatten([FindVariables().visit(var.dimensions) for var in FindVariables().visit(o) if isinstance(var, Array)])

#   remove duplicates from dims
    dims = list(set(dims))

    if return_vars:
        return [var for var in FindVariables().visit(o) if not var in dims]
    else:
        return [var.name for var in FindVariables().visit(o) if not var in dims]
                
def finddimsnotvars(o, return_vars=True):
    """
    Returns only the array dimensions present in an expression or node.
    """

    dims = flatten([FindVariables().visit(var.dimensions) for var in FindVariables().visit(o) if isinstance(var,Array)])

#   remove duplicates from dims
    dims = list(set(dims))

    if return_vars:
        return [var for var in dims]
    else:
        return [var.name for var in dims]           

def loop_check(routine, vars, var_check, output):
    """Checks whether variables with declared intent are used as loop induction variables."""

    for loop in FindNodes(Loop).visit(routine.body):
        for v in [var for var in vars if loop.variable.name == var.name]:
            var_check[v.name][0] = False
            print(f'intent({v.type.intent}) var {v.name} used as loop induction')
            if output:
                with open(output, 'a') as outfile:
                    outfile.write(f'intent({v.type.intent}) var {v.name} used as loop induction\n')

def alloc_check(calls, routine, output):
    """Checks that allocatable variables passed as dummy arguments are not declared intent(out)."""

    alloc_vars = flatten([findvarsnotdims(alloc.variables) for alloc in FindNodes(Allocation).visit(routine.body)])

    for call in calls:
        assert call.routine, f'matching routine for {call} not found'
        for arg, carg in call.arg_iter():
            if getattr(carg, "name", None) in [a.name for a in alloc_vars] and arg.type.intent not in ('in', 'inout'):
                print(f'Allocatable dummy arg {arg} has wrong intent in {call.routine} declaration')
                if output:
                    with open(output, 'a') as outfile:
                        outfile.write(f'Allocatable dummy arg {arg} has wrong intent in {call.routine} declaration\n')

def call_check(calls, routine, output):
    """Checks the consistency of intent declaration across calls to subroutines."""

    assign_type = {var.name: 'none' for var in flatten([[a for f, a in call.arg_iter() if hasattr(a, 'type')] for call in calls])}

    intent_map = {'in': {'none': ['in'], 'lhs': ['in'], 'rhs': ['in']}}
    intent_map['out'] = {'none': ['out'], 'lhs': ['in', 'inout'], 'rhs': ['in', 'inout', 'out']}
    intent_map['inout'] = {'none': ['in', 'inout', 'out'], 'lhs': ['in', 'inout'], 'rhs': ['in', 'inout', 'out']}

    loc = [c for c, n in enumerate([n for n in FindNodes(Node).visit(routine.body) if not isinstance(n,Section)]) if n in calls]

    if loc:
        with dataflow_analysis_attached(routine):
            for n in [n for n in FindNodes(Node).visit(routine.body) if not isinstance(n, Section)][:loc[-1]+1]:
                if n in calls:
                    for f, a in [(f, a) for f, a in n.arg_iter() if getattr(getattr(a, 'type', None), 'intent', None)]:
                        if not f.type.intent in intent_map[a.type.intent][assign_type[a.name]]:
                            print(f'intent inconsistency in {n} for arg {a.name}')
                            if output:
                                with open(output, 'a') as outfile:
                                    outfile.write(f'intent inconsistency in {n} for arg {a.name}\n')
     
                assign_type.update({var: 'rhs' for var in assign_type if var in flatten([findvarsnotdims(s) for s in n.uses_symbols]) and var in flatten([findvarsnotdims(s) for s in n.live_symbols])})
                if isinstance(n, LeafNode):
                    assign_type.update({var: 'lhs' for var in assign_type if var in flatten([findvarsnotdims(s) for s in n.defines_symbols])})

def body_check(routine, in_vars, out_vars, inout_vars, var_check):
    """
    Checks whether the intent of in/out types is violated in the body, and if inout type is used
    only as in.
    """

    with dataflow_analysis_attached(routine):
        # check rules for intent vars in body
        for node in [n for n in FindNodes(Node).visit(routine.body) if not isinstance(n, Section)]:

            for v in [v for v in in_vars if v.name in [f.name.lower() for f in flatten([findvarsnotdims(s) for s in node.uses_symbols])]]:
                var_check[v.name][1] = "Used"

            for v in [v for v in inout_vars if v.name in [f.name.lower() for f in flatten([findvarsnotdims(s) for s in node.uses_symbols])] and var_check[v.name][1] != "Used"]:
                var_check[v.name][1] = "in"
            for v in [v for v in inout_vars if v.name in [f.name.lower() for f in flatten([findvarsnotdims(s) for s in node.defines_symbols])]]:
                var_check[v.name][1] = "Used"

            for v in [v for v in flatten([findvarsnotdims(s) for s in node.defines_symbols]) if v.name.lower() in [v.name for v in out_vars if var_check[v.name][1] == "Unused"]]:
                var_check[v.name][1] = "Used"

            # intent violations across callstatements have already been checked
            if not isinstance(node, CallStatement):
                for v in [v for v in in_vars if v.name in [f.name.lower() for f in flatten([findvarsnotdims(s) for s in node.defines_symbols])]]:
                    var_check[v.name][0] = False
            if isinstance(node, LeafNode) and not isinstance(node, CallStatement):
                for v in [v for v in flatten([findvarsnotdims(s) for s in node.uses_symbols]) if v.name.lower() in [v.name for v in out_vars] and v not in flatten([findvarsnotdims(s) for s in node.live_symbols])]:
                    var_check[v.name][0] = False

def spec_check(routine, vars, var_check, output):
    """
    Check if variables with declared intent are used to define array shape.
    """

    for v in [var for var in vars if var in finddimsnotvars(routine.spec)]:
        if v.type.intent == "in":
            var_check[v.name][1] = "Used"
        elif v.type.intent == "out":
            print(f'intent(out) var {v.name} used in subroutine specification')
            if output:
                with open(output, 'a') as file:
                    file.write(f'intent(out) var {v.name} used in subroutine specification\n')
        else:
            var_check[v.name][1] = "in"

def resolve_member_routine(routine, var_check, in_vars, out_vars, inout_vars):
    """
    Inserting code from internal routine into parent, with declared variabels renamed.
    Variables with intent are added to relevant list, and internal routine is then deleted.
    """

    mem_routines = as_tuple([r for r in routine.contains.body if isinstance(r,Subroutine)])
    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name.name.lower() in str(mem_routines).lower()]

    new_vars=()
    call_map={}
    for n,call in enumerate(calls):

        arg_map = {arg: carg for arg, carg in call.arg_iter()}

        r = call.routine.clone()
        vmap = {}
        for v in [var for var in FindVariables().visit(r.body) if var.name.lower() in [s.name.lower() for s in findvarsnotdims(r.spec)]]:

            if v.name in [arg.name for arg in arg_map]:
                vmap[v] = [carg for arg, carg in arg_map.items() if v.name == arg.name][0].clone()
            else:
                new_vars += as_tuple(v.clone(name=(v.name+f'_{r.name}_tmp_{n}'), scope=routine))
                vmap[v] = new_vars[-1]

        r.body = SubstituteExpressions(vmap).visit(r.body)
        call_map[call] = r.body.clone()

    routine.variables += new_vars
    routine.body = Transformer(call_map).visit(routine.body)
    routine.contains = None

    # add new vars to intent var list
    intent_vars = defaultdict(list)
    for var in [var for var in new_vars if var.type.intent]:
        intent_vars[var.type.intent].update(var)
        var_check[var.name] = [True, "Unused"]

    in_vars += intent_vars['in']
    out_vars += intent_vars['out']
    inout_vars += intent_vars['inout']

def resolve_association(routine):
    """Resolves variable associations and replaces associate block with its body"""

    assoc = FindNodes(Associate).visit(routine.body)[0]
    vmap = {}
    for rexpr,lexpr in assoc.associations:
        vmap.update({var: rexpr.clone(dimensions=getattr(var, "dimensions", None)) for var in [v for v in FindVariables().visit(assoc.body) if lexpr.name.lower() == v.name.lower()]})
    assoc_map = {assoc: SubstituteExpressions(vmap).visit(assoc.body)}
    routine.body = Transformer(assoc_map).visit(routine.body)

def resolve_pointer(routine):
    """Resolves pointer associations and deletes assignment and matching nullify statements"""

    assign = [p for p in FindNodes(Assignment).visit(routine.body) if getattr(p, "ptr", None) and 'null' not in [var.name.lower() for var in findvarsnotdims(getattr(p, "rhs", None))]][0]

    loc = [k for k, n in enumerate(FindNodes(Node).visit(routine.body)) if n == assign]
    assert len(loc)==1, f'location of pointer {assign} not found'

    pointer_map = {assign: None}
    for n in FindNodes(Node).visit(routine.body)[loc[0]+1:]:
        if isinstance(n, Nullify) and assign.lhs in findvarsnotdims(getattr(n, "variables", None)):
            pointer_map[n] = None
            break
        elif isinstance(n, Assignment) and 'null' in [var.name.lower() for var in findvarsnotdims(getattr(n, "rhs", None))]:
            pointer_map[n] = None
            break
                
        vmap = {var: assign.rhs.clone() for var in [v for v in FindVariables().visit(n) if assign.lhs.name == v.name]}
        pointer_map[n] = SubstituteExpressions(vmap).visit(n)

    routine.body = Transformer(pointer_map).visit(routine.body)

def decl_check(routine, output):
    "Checks whether all dummy arguments have a specified intent."

    for v in routine.arguments:
        if not v.type.intent:
            print(f'Dummy argument {v.name} has no declared intent')
            if output:
                with open(output, 'a') as file:
                    file.write(f'Dummy argument {v.name} has no declared intent\n')

def require_output_param(ctx, param, value):
    """Callback function for click that checks the output param is set if summary is set.""" 

    if value:
        if ctx.get_parameter_source('output') == click.core.ParameterSource.DEFAULT:
            ctx.fail('--output must be set in order to generate summary')
    return value

@click.command(help='CLI tool to check if the intent of subroutine dummy arguments is specified correctly.')
@click.option('--mode',default='rule-break',show_default=True,type=click.Choice(['rule-break','rule-unused'],case_sensitive=False),help=('rule-break: check only for intent violations. rule-unused: also check for redundant intent'))
@click.option('--setup',type=click.Choice(['manual','config'],case_sensitive=False),help=("manual: specify sourcefile paths directly. config: read in toml configuration file for the loki scheduler."))
@click.option('--path','-p',type=click.Path(exists=True),multiple=True,help=('Path of source to be parsed or path to configuration file.'))
@click.option('--disable','-d',type=str,multiple=True,default=None,help=('List of function calls to be excluded from intent check. Only used if "--setup manual" specified.'))
@click.option('--output',type=click.Path(),default=None,is_eager=True,help=('Path of file to write output.'))
@click.option('--summary',type=click.Path(),default=None,callback=require_output_param,help=('Path of file to write counted violations. Requires --output option to be set.'))
def check(mode, setup, path, disable, output, summary):
    """
    Program to check if the intent of subroutine dummy arguments is defined correctly
    """

    if summary:
         assert output
    if output:
        with open(output,'w') as outfile:
            outfile.write('')

#   build list of files to be checked and excluded functions
    files = ()
    routines = ()
    if setup == 'manual':

        current_dir = os.getcwd()
        for p in path:
            rel_path = os.path.relpath(p,current_dir)
            files += as_tuple([str(s) for s in Path().glob(rel_path)])

        disable = as_tuple([d.strip('*') for d in disable])

        #  collect all routines
        for file in files:
            print(f'read in File :: {file}')
    
            if output:
                with open(output,'a') as outfile:
                    outfile.write(f'read in File :: {file}\n')
         
            routines += Sourcefile.from_file(file).all_subroutines

    else:
        assert len(path) == 1, f'only one config file should be specified'
        path = path[0]

        with Path(path).open('r') as file:
            config = toml.load(file)
    
        scheduler_config = {'default': config['scheduler_config']}
    
        search_dirs = ()
        searches = config['search']
    
        assert all(search['mode'] == 'select' for search in searches) or all(search['mode'] == 'all' for search in searches)
    
        for search in searches:
           for s in search['dirs']:
               p = Path(PurePath(s))
               p.resolve()
               search_dirs += as_tuple(str(p))
            
        scheduler = Scheduler(paths=search_dirs, config=scheduler_config)
        for search in searches:
            if search['mode'] == 'select':
                r = as_tuple([r.lower() for r in search['routines']])
                scheduler.populate(r)
            else:
                scheduler.populate(list(scheduler.obj_map.keys()))
                break

        scheduler.enrich()
        disable = scheduler.config.disable
        for i in scheduler.items:
            assert i.routine
            routines += as_tuple(i.routine)
            print(f'collected from scheduler {i.routine}')
    
            if output:
                with open(output,'a') as outfile:
                    outfile.write(f'collected from scheduler {i.routine}\n')

    print()
    if output:
        with open(output,'a') as outfile:
            outfile.write('\n')

    if setup == 'manual':
        for routine in routines:
            routine.enrich_calls(routines)

#   main outer loop
    for c, routine in enumerate(routines):
#        if c > 0: break
    
        print(f'checking {routine}')
        if output:
            with open(output,'a') as outfile:
                outfile.write(f'checking {routine}\n')

        # convert entire routine to lowercase
        convert_to_lower_case(routine)
        if routine.contains:
            for node in [node for node in routine.contains.body if isinstance(node, Subroutine)]:
                convert_to_lower_case(node)

        # check if intent is specified for all dummy arguments
        decl_check(routine, output)
        if routine.contains:
            for node in [node for node in routine.contains.body if isinstance(node, Subroutine)]:
                decl_check(node, output)

        # collect variables for which intent is defined
        intent_vars = defaultdict(list)
        for var in routine.arguments:
            intent_vars[var.type.intent].append(var)
        in_vars = intent_vars['in']
        out_vars = intent_vars['out']
        inout_vars = intent_vars['inout']

        # resolving associations in the order in which they appear
        for pointer in FindNodes((Associate, Assignment)).visit(routine.body):
            if isinstance(pointer, Assignment) and getattr(pointer, "ptr", None) and 'null' not in [var.name.lower() for var in findvarsnotdims(getattr(pointer, "rhs", None))]:
                resolve_pointer(routine)
            elif isinstance(pointer, Associate):
                resolve_association(routine)

        # intialize intent rule checks
        var_check = {var.name: [True, "Unused"] for var in in_vars+out_vars+inout_vars}

        # enrich calls for internal routines
        if routine.contains:
            for node in [node for node in routine.contains.body if isinstance(node, Subroutine)]:
                routine.enrich_calls(node)

        # remove specified functions from call list
        calls = [call for call in FindNodes(CallStatement).visit(routine.body)  if not any(d.lower() in call.name.name.lower() for d in disable)]

        #  checking the intent of allocatable vars passed as dummy arguments
        alloc_check(calls, routine, output)
    
        #  checking the intent consistency across function calls
        call_check(calls, routine, output)

####........resolve internal subroutines here........####
        if routine.contains:
            resolve_member_routine(routine, var_check, in_vars, out_vars, inout_vars)

        # check that variables with declared intent aren't used as loop induction variable
        loop_check(routine, in_vars+out_vars+inout_vars, var_check, output)

        # check if intent variables are used to define array shape
        spec_check(routine, in_vars+inout_vars+out_vars, var_check, output)

        # checking intent rules in subroutine body
        body_check(routine, in_vars, out_vars, inout_vars, var_check)

        rule_check(mode, var_check, output, routine)
        print()
        if output:
            with open(output, 'a') as outfile:
                outfile.write('\n')

    if summary:
        count_violations(output, summary)

if __name__ == "__main__":
   check()

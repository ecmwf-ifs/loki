# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.transform import Transformation
from loki.tools import as_tuple
from loki import ir
from loki.subroutine import Subroutine
from loki.module import Module
from loki.expression import Variable, LogicLiteral, IntrinsicLiteral, LogicalNot, LoopRange, SubstituteExpressions
from loki.types import BasicType, SymbolAttributes, ProcedureType, DerivedType

__all__ = ['DerivedTypeDeepcopyTransformation']

class DerivedTypeDeepcopyTransformation(Transformation):


    def __init__(self, **kwargs):
        self.directive = 'openacc'

    @staticmethod
    def create_generic_interface(proc_name, type_name, proc):

        symbol = proc_name + '_' + type_name
        proc_type = SymbolAttributes(dtype=ProcedureType(procedure=proc))
        proc = Variable(name=symbol, type=proc_type)
        proc_decl = ir.ProcedureDeclaration(symbols=as_tuple(proc), module=True)

        intf = ir.Interface(body=as_tuple(proc_decl), spec=proc_name)
        return as_tuple(intf)

    @staticmethod
    def populate_imports(parent):

        util_imports = set('UTIL_' + v.type.dtype.name + '_MOD' for v in parent.variables
                           if isinstance(v.type.dtype, DerivedType))
        return as_tuple([ir.Import(module=util_import) for util_import in util_imports])

    def create_skeleton_subroutine(self, method_name, parent):

        routine_name = method_name + '_' + str(parent.type.dtype).upper()
        routine = Subroutine(name=routine_name, spec=as_tuple(ir.Intrinsic(text='IMPLICIT NONE')),
                             body=as_tuple(ir.Comment('')))

        # define arguments
        routine.arguments = as_tuple(parent)
        if method_name == 'COPY' or method_name == 'WIPE':
            if method_name == 'COPY':
                opt_var = 'LDCREATED'
                local_var = 'LLCREATED'
            else:
                opt_var = 'LDDELETED'
                local_var = 'LLDELETED'
            ldmethod = Variable(name=opt_var, type=SymbolAttributes(dtype=BasicType.LOGICAL,
                                                                     optional=True, intent='IN'))

            routine.arguments += as_tuple(ldmethod)

            # define local vars
            llmethod = Variable(name=local_var, type=SymbolAttributes(dtype=BasicType.LOGICAL))
            routine.variables += as_tuple(llmethod)

            # add LDCREATED/LDDELETED check
            routine.body.append(as_tuple(ir.Assignment(lhs=llmethod, rhs=LogicLiteral(value=False))))
            routine.body.append(as_tuple(self.arg_present_check(ldmethod, llmethod)))
            routine.body.append(as_tuple(ir.Comment('')))

        # populate necessary util module imports
        util_imports = self.populate_imports(parent)
        routine.spec.prepend(util_imports)

        return routine

    @staticmethod
    def arg_present_check(arg, local):

        condition = IntrinsicLiteral(value=f'PRESENT({arg.name.upper()})')
        assignment = ir.Assignment(lhs=local, rhs=arg)

        return ir.Conditional(condition=condition, body=as_tuple(assignment))

    @staticmethod
    def create_util_call(action, var, parent, routine):

        kwargs = ()
        if action == 'COPY':
            kwargs += (('LDCREATED', LogicLiteral(value=True)),)
        elif action == 'WIPE':
            kwargs += (('LDDELETED', LogicLiteral(value=True)),)

        call_name = Variable(name=action + '_' + str(var.type.dtype).upper())

        if var.type.shape:
            loopbody = ()
            for dim in range(len(var.type.shape)):
                routine.variables += as_tuple(Variable(name=f'J{dim+1}', type=SymbolAttributes(dtype=BasicType.INTEGER)))

                loopvar = Variable(name=f'J{dim+1}', type=SymbolAttributes(dtype=BasicType.INTEGER))
                lstart = IntrinsicLiteral(value=f'LBOUND({parent.name}%{var.name},{dim+1})')
                lend = IntrinsicLiteral(value=f'UBOUND({parent.name}%{var.name},{dim+1})')
                bounds = LoopRange((lstart, lend))

                if not loopbody:
                    args = as_tuple(Variable(name=var.name, parent=parent, type=var.type,
                                             dimensions=Variable(name=f'J{dim+1}')))
                    loopbody = as_tuple(ir.CallStatement(name=call_name, arguments=args, kwarguments=kwargs))
                else:
                    vmap = {Variable(name=f'J{dim}'): Variable(name=f'J{dim}, J{dim+1}')}
                    loopbody = as_tuple(SubstituteExpressions(vmap).visit(loopbody))

                loop = ir.Loop(variable=loopvar, bounds=bounds, body=loopbody)
                loopbody = loop

            body = as_tuple(loop)
        else:
            args = as_tuple(Variable(name=var.name, parent=parent, type=var.type))
            body = as_tuple(ir.CallStatement(name=call_name, arguments=args, kwarguments=kwargs))

        return body

    @staticmethod
    def create_memory_status_test(check, var, parent, body):

        condition = IntrinsicLiteral(value=f'{check}({parent.name}%{var.name})')
        return as_tuple(ir.Conditional(condition=condition, body=body))

    def create_copy_method(self, parent_type):

        _derived_type = SymbolAttributes(dtype=parent_type.dtype, intent='INOUT')
        yd = Variable(name='YD', type=_derived_type)
        routine = self.create_skeleton_subroutine('COPY', yd)

        # create YD
        llcreated = Variable(name='LLCREATED', type=SymbolAttributes(dtype=BasicType.LOGICAL))
        condition = LogicalNot(llcreated)
        if self.directive == 'openacc':
            body = as_tuple(ir.Pragma(keyword='acc', content=f'enter data copyin({yd.name})'))
        routine.body.append(as_tuple(ir.Conditional(condition=condition, body=body)))
        routine.body.append(as_tuple(ir.Comment('')))

        for var in parent_type.variables:

            instr = ()
            if isinstance(var.type.dtype, DerivedType):
                instr += self.create_util_call('COPY', var, yd, routine)

            if var.type.allocatable or var.type.pointer:
                if var.type.allocatable:
                    check = 'ALLOCATED'
                else:
                    check = 'ASSOCIATED'

                if self.directive == 'openacc':
                    pragma = ir.Pragma(keyword='acc', content=f'enter data copyin({yd.name}%{var.name})')
                instr = (pragma, *instr)
                instr = self.create_memory_status_test(check, var, yd, body=instr)

            if instr:
                routine.body.append(instr)
                routine.body.append(as_tuple(ir.Comment('')))


        return routine

    def create_host_method(self, parent_type):

        _derived_type = SymbolAttributes(dtype=parent_type.dtype, intent='INOUT')
        yd = Variable(name='YD', type=_derived_type)
        routine = self.create_skeleton_subroutine('HOST', yd)

        for var in parent_type.variables:

            instr = ()
            if isinstance(var.type.dtype, DerivedType):
                instr = self.create_util_call('HOST', var, yd, routine)
            elif isinstance(var.type.dtype, BasicType):
                if self.directive == 'openacc':
                    instr = as_tuple(ir.Pragma(keyword='acc', content=f'exit data copyout({yd.name}%{var.name})'))

            if var.type.allocatable or var.type.pointer:
                if var.type.allocatable:
                    check = 'ALLOCATED'
                else:
                    check = 'ASSOCIATED'

                instr = self.create_memory_status_test(check, var, yd, body=instr)

            if instr:
                routine.body.append(instr)
                routine.body.append(as_tuple(ir.Comment('')))


        return routine

    def create_wipe_method(self, parent_type):

        _derived_type = SymbolAttributes(dtype=parent_type.dtype, intent='INOUT')
        yd = Variable(name='YD', type=_derived_type)
        routine = self.create_skeleton_subroutine('WIPE', yd)

        for var in parent_type.variables:

            instr = ()
            if isinstance(var.type.dtype, DerivedType):
                instr += self.create_util_call('WIPE', var, yd, routine)

            if var.type.allocatable or var.type.pointer:
                if var.type.allocatable:
                    check = 'ALLOCATED'
                else:
                    check = 'ASSOCIATED'

                if self.directive == 'openacc':
                    pre_pragma = ir.Pragma(keyword='acc', content=f'exit data detach({yd.name}%{var.name})')
                    post_pragma = ir.Pragma(keyword='acc', content=f'exit data delete({yd.name}%{var.name})')
                instr = (pre_pragma, *instr, post_pragma)
                instr = self.create_memory_status_test(check, var, yd, body=instr)

            if instr:
                routine.body.append(instr)
                routine.body.append(as_tuple(ir.Comment('')))

        # delete YD
        lldeleted = Variable(name='LLDELETED', type=SymbolAttributes(dtype=BasicType.LOGICAL))
        condition = LogicalNot(lldeleted)
        if self.directive == 'openacc':
            body = as_tuple(ir.Pragma(keyword='acc', content=f'exit data delete({yd.name})'))
        routine.body.append(as_tuple(ir.Conditional(condition=condition, body=body)))
        routine.body.append(as_tuple(ir.Comment('')))

        return routine

    def create_module(self, parent_module_name, parent_type):

        # assemble module name
        name = 'UTIL' + '_' + parent_type.name.upper() + '_MOD'

        # import parent_type
        spec = as_tuple(ir.Import(module=parent_module_name, symbols=as_tuple(Variable(name=parent_type.name))))

        # create offload methods
        copy_method = self.create_copy_method(parent_type)
        host_method = self.create_host_method(parent_type)
        wipe_method = self.create_wipe_method(parent_type)

        # define overloaded interfaces for contained methods
        spec += as_tuple(ir.Comment(''))
        spec += self.create_generic_interface('COPY', parent_type.name, copy_method)
        spec += self.create_generic_interface('HOST', parent_type.name, host_method)
        spec += self.create_generic_interface('WIPE', parent_type.name, wipe_method)
        spec += as_tuple(ir.Comment(''))

        contains = (ir.Comment(''), copy_method)
        contains += (ir.Comment(''), host_method)
        contains += (ir.Comment(''), wipe_method)

        return Module(name=name, spec=spec, contains=contains)

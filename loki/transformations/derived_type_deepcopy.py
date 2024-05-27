# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from loki.tools import as_tuple
from loki.ir import nodes as ir
from loki.subroutine import Subroutine
from loki.module import Module
from loki.expression import Variable, LogicLiteral, IntrinsicLiteral, LogicalNot, LoopRange, SubstituteExpressions
from loki.expression import IntLiteral, RangeIndex, ProcedureSymbol, InlineCall
from loki.types import BasicType, SymbolAttributes, ProcedureType, DerivedType
from loki.batch import Transformation, TypeDefItem
from loki.sourcefile import Sourcefile

__all__ = ['DerivedTypeDeepcopyTransformation']

class DerivedTypeDeepcopyTransformation(Transformation):

    item_filter = (TypeDefItem)
    process_ignored_items = True

    def __init__(self, builddir):
        self.directive = 'openacc'
        self.builddir = builddir

    @staticmethod
    def create_generic_interface(proc_name, type_name, proc):

        symbol = proc_name + '_' + type_name
        proc_type = SymbolAttributes(dtype=ProcedureType(procedure=proc))
        proc = Variable(name=symbol, type=proc_type)
        proc_decl = ir.ProcedureDeclaration(symbols=as_tuple(proc), module=True)

        intf = ir.Interface(body=as_tuple(proc_decl), spec=proc_name)
        return as_tuple(intf)

    @staticmethod
    def populate_imports(parent, successor_types):

        util_imports = set('UTIL_' + v.type.dtype.name + '_MOD' for v in parent.variables
                           if isinstance(v.type.dtype, DerivedType) and v.type.dtype in successor_types)
        return as_tuple([ir.Import(module=util_import) for util_import in util_imports])

    def create_skeleton_subroutine(self, method_name, _type, successor_types):

        routine_name = method_name + '_' + str(_type.dtype).upper()
        routine = Subroutine(name=routine_name, spec=as_tuple(ir.Intrinsic(text='IMPLICIT NONE')),
                             body=as_tuple(ir.Comment('')))

        # define arguments
        yd = Variable(name='YD', type=_type, scope=routine)
        routine.arguments = as_tuple(yd)
        if method_name == 'COPY' or method_name == 'WIPE':
            if method_name == 'COPY':
                opt_var = 'LDCREATED'
                local_var = 'LLCREATED'
            else:
                opt_var = 'LDDELETED'
                local_var = 'LLDELETED'
            ldmethod = Variable(name=opt_var, type=SymbolAttributes(dtype=BasicType.LOGICAL,
                                                                     optional=True, intent='IN'), scope=routine)

            routine.arguments += as_tuple(ldmethod)

            # define local vars
            llmethod = Variable(name=local_var, type=SymbolAttributes(dtype=BasicType.LOGICAL), scope=routine)
            routine.variables += as_tuple(llmethod)

            # add LDCREATED/LDDELETED check
            routine.body.append(as_tuple(ir.Assignment(lhs=llmethod, rhs=LogicLiteral(value=False))))
            routine.body.append(as_tuple(self.arg_present_check(ldmethod, llmethod)))
            routine.body.append(as_tuple(ir.Comment('')))

        # populate necessary util module imports
        util_imports = self.populate_imports(yd, successor_types)
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

    @staticmethod
    def create_field_api_call(field_object, dest, access, ptr):
        return ir.CallStatement(name=Variable(name='GET_' + dest.upper() + '_DATA_' + access, parent=field_object),
                                arguments=as_tuple(ptr))

    @staticmethod
    def create_aliased_ptr_assignment(ptr, alias, yd):
        dims = [InlineCall(function=ProcedureSymbol('LBOUND', scope=yd.scope), parameters=(ptr, IntLiteral(r+1)))
                for r in range(len(ptr.shape))]
        lhs = ptr.parent.type.dtype.typedef.variable_map[alias].clone(parent=yd,
            dimensions=as_tuple([RangeIndex(children=(d, None)) for d in dims]))
        return ir.Assignment(lhs=lhs, rhs=ptr, ptr=True)

    def create_field_api_copy(self, var, aliased_ptrs, yd):
        field_ptr_name = var.name[1:] + '_FIELD'
        field_object = var.clone(parent=yd)
        field_ptr_var = var.scope.variable_map[field_ptr_name].clone(parent=yd, dimensions=None)
        body = as_tuple(self.create_field_api_call(field_object, 'DEVICE', 'RDWR',
                                                   field_ptr_var))
        if self.directive == 'openacc':
            body += as_tuple(ir.Pragma(keyword='acc', content=f'enter data attach({field_ptr_var})'))
        if field_ptr_name.lower() in aliased_ptrs:
            body += as_tuple(self.create_aliased_ptr_assignment(field_ptr_var,
                                                                aliased_ptrs[field_ptr_name.lower()], yd))
            if self.directive == 'openacc':
                body += as_tuple(ir.Pragma(keyword='acc',
                                           content=f'enter data attach({body[-1].lhs.name})'))

        return body

    def create_field_api_host(self, var, aliased_ptrs, yd):
        field_ptr_name = var.name[1:] + '_FIELD'
        field_object = var.clone(parent=yd)
        field_ptr_var = var.scope.variable_map[field_ptr_name].clone(parent=yd, dimensions=None)
        body = as_tuple(self.create_field_api_call(field_object, 'HOST', 'RDWR',
                                                   field_ptr_var))
        if field_ptr_name.lower() in aliased_ptrs:
            body += as_tuple(self.create_aliased_ptr_assignment(field_ptr_var,
                                                                aliased_ptrs[field_ptr_name.lower()], yd))
        return body

    def create_field_api_wipe(self, var, aliased_ptrs, yd):
        field_ptr_name = var.name[1:] + '_FIELD'
        field_ptr_var = var.scope.variable_map[field_ptr_name].clone(parent=yd, dimensions=None)
        field_object = var.clone(parent=yd)
        body = ()
        if field_ptr_name.lower() in aliased_ptrs:
            if self.directive == 'openacc':
                arg = var.scope.variable_map[aliased_ptrs[field_ptr_name.lower()]].clone(parent=yd, dimensions=None)
                body += as_tuple(ir.Pragma(keyword='acc',
                                           content=f'exit data detach({arg})'))
        if self.directive == 'openacc':
            body += as_tuple(ir.Pragma(keyword='acc', content=f'exit data detach({field_ptr_var})'))
            body += as_tuple(ir.CallStatement(name=Variable(name='DELETE_DEVICE_DATA', parent=field_object),
                                     arguments=()))
        return body

    def create_copy_method(self, parent_type, offload_vars, aliased_ptrs, successor_types):

        _derived_type = SymbolAttributes(dtype=parent_type.dtype, intent='INOUT')
        routine = self.create_skeleton_subroutine('COPY', _derived_type, successor_types)
        yd = routine.variable_map['yd']

        # create YD
        llcreated = Variable(name='LLCREATED', type=SymbolAttributes(dtype=BasicType.LOGICAL))
        condition = LogicalNot(llcreated)
        if self.directive == 'openacc':
            body = as_tuple(ir.Pragma(keyword='acc', content=f'enter data copyin({yd.name})'))
        routine.body.append(as_tuple(ir.Conditional(condition=condition, body=body)))
        routine.body.append(as_tuple(ir.Comment('')))

        for var in offload_vars:

            if var.type.allocatable:
                check = 'ALLOCATED'
            else:
                check = 'ASSOCIATED'

            instr = ()
            if (result := re.match(r'FIELD_[1-5][a-z]{2}', var.type.dtype.name, re.IGNORECASE)):
                instr += self.create_field_api_copy(var, aliased_ptrs, yd)
                instr = self.create_memory_status_test(check, var, yd, body=instr)
            elif isinstance(var.type.dtype, DerivedType):
                instr += self.create_util_call('COPY', var, yd, routine)

            if var.type.allocatable or var.type.pointer and not result:
                if self.directive == 'openacc':
                    pragma = ir.Pragma(keyword='acc', content=f'enter data copyin({yd.name}%{var.name})')
                instr = (pragma, *instr)
                instr = self.create_memory_status_test(check, var, yd, body=instr)

            if instr:
                routine.body.append(instr)
                routine.body.append(as_tuple(ir.Comment('')))


        return routine

    def create_host_method(self, parent_type, offload_vars, aliased_ptrs, successor_types):

        _derived_type = SymbolAttributes(dtype=parent_type.dtype, intent='INOUT')
        routine = self.create_skeleton_subroutine('HOST', _derived_type, successor_types)
        yd = routine.variable_map['yd']

        for var in offload_vars:

            if var.type.allocatable:
                check = 'ALLOCATED'
            else:
                check = 'ASSOCIATED'

            instr = ()
            if (result := re.match(r'FIELD_[1-5][a-z]{2}', var.type.dtype.name, re.IGNORECASE)):
                instr += self.create_field_api_host(var, aliased_ptrs, yd)
                instr = self.create_memory_status_test(check, var, yd, body=instr)
            elif isinstance(var.type.dtype, DerivedType):
                instr = self.create_util_call('HOST', var, yd, routine)
            elif isinstance(var.type.dtype, BasicType):
                if self.directive == 'openacc':
                    instr = as_tuple(ir.Pragma(keyword='acc', content=f'exit data copyout({yd.name}%{var.name})'))

            if var.type.allocatable or var.type.pointer and not result:
                instr = self.create_memory_status_test(check, var, yd, body=instr)

            if instr:
                routine.body.append(instr)
                routine.body.append(as_tuple(ir.Comment('')))


        return routine

    def create_wipe_method(self, parent_type, offload_vars, aliased_ptrs, successor_types):

        _derived_type = SymbolAttributes(dtype=parent_type.dtype, intent='INOUT')
        routine = self.create_skeleton_subroutine('WIPE', _derived_type, successor_types)
        yd = routine.variable_map['yd']

        for var in offload_vars:

            if var.type.allocatable:
                check = 'ALLOCATED'
            else:
                check = 'ASSOCIATED'

            instr = ()
            if (result := re.match(r'FIELD_[1-5][a-z]{2}', var.type.dtype.name, re.IGNORECASE)):
                instr += self.create_field_api_wipe(var, aliased_ptrs, yd)
                instr = self.create_memory_status_test(check, var, yd, body=instr)
            elif isinstance(var.type.dtype, DerivedType):
                instr += self.create_util_call('WIPE', var, yd, routine)

            if var.type.allocatable or var.type.pointer and not result:
                if self.directive == 'openacc':
                    pragma = ir.Pragma(keyword='acc', content=f'exit data delete({yd.name}%{var.name})')
                instr = (*instr, pragma)
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

    def create_module(self, parent_module_name, parent_type, offload_vars, aliased_ptrs, successor_types):

        # assemble module name
        name = 'UTIL' + '_' + parent_type.name.upper() + '_MOD'

        # import parent_type
        spec = as_tuple(ir.Import(module=parent_module_name, symbols=as_tuple(Variable(name=parent_type.name))))

        # create offload methods
        copy_method = self.create_copy_method(parent_type, offload_vars, aliased_ptrs, successor_types)
        host_method = self.create_host_method(parent_type, offload_vars, aliased_ptrs, successor_types)
        wipe_method = self.create_wipe_method(parent_type, offload_vars, aliased_ptrs, successor_types)

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

    def transform_typedef(self, typedef, **kwargs):

        if 'cpg_sl1_3d' in typedef.name.lower() or 'cpg_sl1_2d' in typedef.name.lower():
            return

        item = kwargs['item']
        successors = kwargs.get('successors', [])
        skip_offload = item.config.get('skip_offload', [])
        aliased_ptrs = item.config.get('aliased_ptrs', {})

        successor_types = [item.ir.dtype for item in successors]
        offload_vars = [v for v in typedef.variables if not v.name.lower() in skip_offload]

        module = self.create_module(typedef.parent.name, typedef, offload_vars, aliased_ptrs, successor_types)
        fnm = self.builddir + '/' + module.name
        source = Sourcefile(path=fnm.lower() + '.F90', ir=as_tuple(module))
        source.write()

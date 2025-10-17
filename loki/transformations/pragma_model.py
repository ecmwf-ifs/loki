# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect

from loki.batch import Transformation, ProcedureItem, ModuleItem
from loki.ir import (
    FindNodes, Pragma, Transformer, get_pragma_command_and_parameters
)

__all__ = ['PragmaModelTransformation']


class GenericPragmaMapper:
    """
    A generic pragma mapper class.

    Pragmas in the form

    ``!$loki command-optionally-with-hyphen [param] [param_with_val(val)]``

    get a visitor/handler method that looks like

    .. code-block::
       def visit_command_optionally_with_hyphen(self, pragma, [**kwargs]):
           pass

    The handler is responsible for returning either None or the updated
    pragma.
    """
    # pylint: disable=unused-argument
    def __init__(self):
        handlers = {}
        prefix = "pmap_"
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            argspec = inspect.getfullargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "pmap_foo(self, pragma, [**kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    def lookup_method(self, starts_with):
        try:
            return self._handlers[starts_with.lower()]
        except KeyError:
            return None

    @classmethod
    def default_retval(cls):
        """
        Default return value for handler methods.

        Returns
        -------
        None
        """
        return None

    def pmap(self, pragma, **kwargs):
        starts_with, parameters = get_pragma_command_and_parameters(pragma)
        meth = self.lookup_method(starts_with.lower().replace('-', '_'))
        if meth is not None:
            return meth(pragma, parameters, **kwargs)
        return self.default_retval()


class OpenACCPragmaMapper(GenericPragmaMapper):
    """
    Loki generic pragmas to OpenACC mapper.
    """
    # pylint: disable=unused-argument
    def pmap_create(self, pragma, parameters, **kwargs):
        if param_device := parameters.get('device'):
            return Pragma(keyword='acc', content=f'declare create({param_device})')
        return self.default_retval()

    def pmap_update(self, pragma, parameters, **kwargs):
        content = ''
        if param_device := parameters.get('device'):
            content += f' device({param_device})'
        if param_host := parameters.get('host'):
            content += f' self({param_host})'
        if content:
            return Pragma(keyword='acc', content=f'update{content}')
        return self.default_retval()

    def pmap_unstructured_data(self, pragma, parameters, **kwargs):
        content = ''
        if param_in := parameters.get('in'):
            content += f' copyin({param_in})'
        if param_create := parameters.get('create'):
            content += f' create({param_create})'
        if param_attach := parameters.get('attach'):
            content += f' attach({param_attach})'
        if content:
            return Pragma(keyword='acc', content=f'enter data{content}')
        return self.default_retval()

    def pmap_exit_unstructured_data(self, pragma, parameters, **kwargs):
        content = ''
        if params_out := parameters.get('out'):
            content += f' copyout({params_out})'
        if params_delete := parameters.get('delete'):
            content += f' delete({params_delete})'
        if param_detach := parameters.get('detach'):
            content += f' detach({param_detach})'
        if content:
            # Rather than simply decrementing the dynamic reference counter,
            # finalize forces it to zero. This isn't needed for OpenMP, where
            # target exit data map(delete:<>) statement already sets the
            # dynamic reference counter to 0
            final = ' finalize' if 'finalize' in parameters else ''
            return Pragma(keyword='acc', content=f'exit data{content}{final}')
        return self.default_retval()

    def pmap_structured_data(self, pragma, parameters, **kwargs):
        content = ''
        if params_in := parameters.get('in'):
            content += f' copyin({params_in})'
        if params_inout := parameters.get('inout'):
            content += f' copy({params_inout})'
        if params_out := parameters.get('out'):
            content += f' copyout({params_out})'
        if params_create := parameters.get('create'):
            content += f' create({params_create})'
        if params_default := parameters.get('default'):
            content += f' default({params_default})'
        if params_default := parameters.get('present'):
            content += f' present({params_default})'
        if params_asynchronous := parameters.get('async'):
            content += f' async({params_asynchronous})'
        if content:
            return Pragma(keyword='acc', content=f'data{content}')
        return self.default_retval()

    def pmap_end_structured_data(self, pragma, parameters, **kwargs):
        return Pragma(keyword='acc', content='end data')

    def pmap_routine(self, pragma, parameters, **kwargs):
        if 'seq' in parameters:
            return Pragma(keyword='acc', content='routine seq')
        if 'vector' in parameters:
            return Pragma(keyword='acc', content='routine vector')
        return self.default_retval()

    def pmap_loop(self, pragma, parameters, **kwargs):
        if 'seq' in parameters:
            return Pragma(keyword='acc', content='loop seq')
        if 'vector' in parameters:
            private_param = parameters.get('private')
            private = f' private({private_param})' if private_param else ''
            fprivate_param = parameters.get('firstprivate')
            fprivate = f' firstprivate({fprivate_param})' if fprivate_param else ''
            reduction_param = parameters.get('reduction')
            reduction = f' reduction({reduction_param})' if reduction_param else ''
            content = f'loop vector{private}{fprivate}{reduction}'
            return Pragma(keyword='acc', content=content)
        if 'gang' in parameters:
            private_param = parameters.get('private')
            private = f' private({private_param})' if private_param else ''
            fprivate_param = parameters.get('firstprivate')
            fprivate = f' firstprivate({fprivate_param})' if fprivate_param else ''
            vlength_param = parameters.get('vlength')
            vlength = f' vector_length({vlength_param})' if vlength_param else ''
            asynchronous_param = parameters.get('async')
            asynchronous = f' async({asynchronous_param})' if asynchronous_param else ''
            content = f'parallel loop gang{private}{fprivate}{vlength}{asynchronous}'
            return Pragma(keyword='acc', content=content)
        return self.default_retval()

    def pmap_end_loop(self, pragma, parameters, **kwargs):
        if 'gang' in parameters:
            return Pragma(keyword='acc', content='end parallel loop')
        return self.default_retval()

    def pmap_device_present(self, pragma, parameters, **kwargs):
        asynchronous_param = parameters.get('async')
        asynchronous = f' async({asynchronous_param})' if asynchronous_param else ''
        if param_vars := parameters.get('vars'):
            return Pragma(keyword='acc', content=f'data present({param_vars})'+asynchronous)
        return self.default_retval()

    def pmap_end_device_present(self, pragma, parameters, **kwargs):
        return Pragma(keyword='acc', content='end data')

    def pmap_device_ptr(self, pragma, parameters, **kwargs):
        asynchronous_param = parameters.get('async')
        asynchronous = f' async({asynchronous_param})' if asynchronous_param else ''
        if param_vars := parameters.get('vars'):
            return Pragma(keyword='acc', content=f'data deviceptr({param_vars})'+asynchronous)
        return self.default_retval()

    def pmap_end_device_ptr(self, pragma, parameters, **kwargs):
        return Pragma(keyword='acc', content='end data')


class OpenMPOffloadPragmaMapper(GenericPragmaMapper):
    """
    Loki generic pragmas to OpenMP offload/GPU mapper.

    TODO: this is not yet complete!
    """
    # pylint: disable=unused-argument
    def pmap_create(self, pragma, parameters, **kwargs):
        if param_device := parameters.get('device'):
            return Pragma(keyword='omp', content=f'declare target({param_device})')
        return self.default_retval()

    def pmap_update(self, pragma, parameters, **kwargs):
        content = ''
        if param_device := parameters.get('device'):
            content += f' to({param_device})'
        if param_host := parameters.get('host'):
            content += f' from({param_host})'
        if content:
            return Pragma(keyword='omp', content=f'target update{content}')
        return self.default_retval()

    def pmap_unstructured_data(self, pragma, parameters, **kwargs):
        content = ''
        if param_in := parameters.get('in'):
            content += f' map(to: {param_in})'
        if param_create := parameters.get('create'):
            content += f' map(alloc: {param_create})'
        if content:
            return Pragma(keyword='omp', content=f'target enter data{content}')
        return self.default_retval()

    def pmap_exit_unstructured_data(self, pragma, parameters, **kwargs):
        content = ''
        if params_out := parameters.get('out'):
            content += f' map(from: {params_out})'
        if params_delete := parameters.get('delete'):
            content += f' map(delete: {params_delete})'
        if content:
            return Pragma(keyword='omp', content=f'target exit data{content}')
        return self.default_retval()

    def pmap_structured_data(self, pragma, parameters, **kwargs):
        content = ''
        params_in = parameters.get('in', None)
        params_present = parameters.get('present', None)
        # both 'in'/'copyin' and 'present' map to 'map(to: ...)'
        if params_in is not None and params_present is not None:
            content += f' map(to: {params_in}, {params_present})'
        else:
            if params_in is not None:
                content += f' map(to: {params_in})'
            if params_present is not None:
                content += f' map(to: {params_present})'
        if params_inout := parameters.get('inout'):
            content += f' map(tofrom: {params_inout})'
        if params_out := parameters.get('out'):
            content += f' map(from: {params_out})'
        if params_create := parameters.get('create'):
            content += f' map(alloc: {params_create})'
        if content:
            return Pragma(keyword='omp', content=f'target data{content}')
        return self.default_retval()

    def pmap_end_structured_data(self, pragma, parameters, **kwargs):
        return Pragma(keyword='omp', content='end target data')

    def pmap_routine(self, pragma, parameters, **kwargs):
        if 'seq' in parameters:
            return Pragma(keyword='omp', content='declare target')
        return self.default_retval()

    def pmap_loop(self, pragma, parameters, **kwargs):
        if 'vector' in parameters:
            # TODO: private and reduction clause?
            content = 'parallel do'
            return Pragma(keyword='omp', content=content)
        if 'gang' in parameters:
            # TODO: private clause?
            vlength_param = parameters.get('vlength')
            vlength = f' thread_limit({vlength_param})' if vlength_param else ''
            content = f'target teams distribute{vlength}'
            return Pragma(keyword='omp', content=content)
        return self.default_retval()

    def pmap_end_loop(self, pragma, parameters, **kwargs):
        if 'vector' in parameters:
            return Pragma(keyword='omp', content='end parallel do')
        if 'gang' in parameters:
            return Pragma(keyword='omp', content='end target teams distribute')
        return self.default_retval()

    def pmap_omp_update_global_vars(self, pragma, parameters, **kwargs):
        # this shouldn't be necessary but is currently necessary because of a bug in OpenMP
        if params_in := parameters.get('in'):
            return Pragma(keyword='omp', content=f'target enter data map(to: {params_in})')
        return self.default_retval()



class OpenMPThreadingPragmaMapper(GenericPragmaMapper):
    """
    Loki generic pragmas to OpenMP CPU mapper.

    TODO: this is obviously incomplete!
    """
    # pylint: disable=unused-argument
    def pmap_loop(self, pragma, parameters, **kwargs):
        if 'gang' in parameters:
            private_param = parameters.get('private')
            private = f' private({private_param})' if private_param else ''
            fprivate_param = parameters.get('firstprivate')
            fprivate = f' firstprivate({fprivate_param})' if fprivate_param else ''
            default_param = parameters.get('default')
            default = f' default({default_param})' if default_param else ''
            content = f'parallel do {default}{private}{fprivate}'
            return Pragma(keyword='omp', content=content)
        return self.default_retval()


class PragmaModelTransformation(Transformation):
    """
    Transformation to map Loki generic pragmas to a specific
    pragma model using a child class of :any:`GenericPragmaMapper`.

    For the mapping between Loki directives and programming model-specific annotations,
    see :ref:`programming_models:Loki directives`.

    Parameters
    ----------
    directive : False, str
        The directive(s) to be used, used to determine which
        child class of :any:`GenericPragmaMapper` is used.  Use
        ``False`` to suppress the directive translation entirely.
    keep_loki_pragmas: bool
        Keep or remove generic Loki pragmas that are not
        mapped.
    """
    item_filter = (ProcedureItem, ModuleItem)

    def __init__(self, directive=False, keep_loki_pragmas=True):
        assert directive in [False, 'openacc', 'omp-gpu', 'openmp']
        self.directive = directive
        self.keep_loki_pragmas = keep_loki_pragmas
        pmapper_cls_map = {
            'openacc': OpenACCPragmaMapper,
            'omp-gpu': OpenMPOffloadPragmaMapper,
            'openmp': OpenMPThreadingPragmaMapper,
        }
        pmapper_cls = pmapper_cls_map.get(self.directive, None if self.keep_loki_pragmas else GenericPragmaMapper)
        self.pmapper = pmapper_cls() if pmapper_cls else None

    def _create_pragma_map(self, pragmas):
        pragma_map = {}
        for pragma in pragmas:
            new_pragma = self.pmapper.pmap(pragma)
            # either keep loki pragmas that do not have a mapping
            if self.keep_loki_pragmas:
                if new_pragma is not None:
                    pragma_map[pragma] = new_pragma
            # or remove, since pmap returns None
            else:
                pragma_map[pragma] = new_pragma
        return pragma_map

    def transform_module(self, module, **kwargs):
        if self.pmapper is None:
            return
        loki_pragmas = [pragma for pragma in FindNodes(Pragma).visit(module.spec) if pragma.keyword.lower() == 'loki']
        pragma_map = self._create_pragma_map(loki_pragmas)

        module.spec = Transformer(pragma_map).visit(module.spec)

    def transform_subroutine(self, routine, **kwargs):
        if self.pmapper is None:
            return
        loki_pragmas = [pragma for pragma in FindNodes(Pragma).visit(routine.ir) if pragma.keyword.lower() == 'loki']
        pragma_map = self._create_pragma_map(loki_pragmas)

        routine.spec = Transformer(pragma_map).visit(routine.spec)
        routine.body = Transformer(pragma_map).visit(routine.body)

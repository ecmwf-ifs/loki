from loki import ir
from loki.visitors import Transformer, FindNodes
from loki.transform.transformation import Transformation


__all__ = ['HoistAnalysis', 'HoistSynthesis']


class HoistAnalysis(Transformation):
    """
    Analysis part of the hoist functionality/transformation.
    To be applied reversed, in order to recursively find all variables to be hoisted.
    """

    def __init__(self):
        pass

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs.get('role', None)
        item = kwargs.get('item', None)
        successors = kwargs.get('successors', None)

        if item and not item.local_name == routine.name.lower():
            return

        variables = self.find_variables(routine)
        if role != 'driver':
            item.user_data["to_hoist"] = variables
            item.user_data["hoist_variables"] = [var.clone(name=f'{routine.name}_{var.name}') for var in variables]
        else:
            item.user_data["to_hoist"] = []
            item.user_data["hoist_variables"] = []

        for child in successors:
            item.user_data["to_hoist"].extend(child.user_data["hoist_variables"])
            item.user_data["hoist_variables"].extend(child.user_data["hoist_variables"])

    @staticmethod
    def find_variables(routine):
        return [var for var in routine.variables if var not in routine.arguments if not var.type.parameter]


class HoistSynthesis(Transformation):
    """
    Synthesis part of the hoist functionality/transformation.
    **Needs the `HoistAnalysis` part to be processed first** in order to hoist all already found variables.
    """

    def __init__(self):
        pass

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role', None)
        item = kwargs.get('item', None)
        successors = kwargs.get('successors', None)
        successor_map = {successor.routine.name: successor for successor in successors}

        if item and not item.local_name == routine.name.lower():
            return

        if role == 'driver':
            for var in item.user_data["to_hoist"]:
                self.driver_variable_declaration(routine, var)
        else:
            routine.arguments += ir.as_tuple([var.clone(type=var.type.clone(intent='inout', allocatable=None))
                                           for var in item.user_data["to_hoist"]])

        call_map = {}
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            new_args = [arg.clone(dimensions=None) for arg in successor_map[call.name].user_data["hoist_variables"]]
            arguments = list(call.arguments) + new_args
            call_map[call] = call.clone(arguments=ir.as_tuple(arguments))

        routine.body = Transformer(call_map).visit(routine.body)

    @staticmethod
    def driver_variable_declaration(routine, var):
        routine.spec.append(ir.VariableDeclaration((var,)))

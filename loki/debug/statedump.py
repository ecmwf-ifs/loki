"""
Utility sub-package for generating state dumps from Fortran code.

The Fortran utility module ``loki_debug`` provides a Fortran API
to generate state dumps for multiple variables that can then be
loaded into Python via the provided :class:`StateDump` class.
Some utility routines to generate the boilerplate dumping code
are also provided.
"""
import numpy as np
from collections import OrderedDict, defaultdict
from pathlib import Path
from csv import reader as CSVReader
from functools import reduce
from operator import mul


__all__ = ['StateDump']


class StateDump:
    """
    Collection of variables and arrays to represent a dumped state
    as created by the `loki_debug::StateDump` utility.
    """

    # Type encoding for the binary data read
    _readtype = {
        'int32': 'i4', 'float64': 'f8', 'logical': 'i4'
    }

    # Numpy data type to allocate arrays
    _alloctype = {
        'int32': np.int32, 'float64': np.float64, 'logical': np.bool
    }

    @classmethod
    def template_dump_state(variables):
        """
        Generate boilerplate state_dump code from list of variables
        """
        print('    allocate(items(%d))' % len(variables))
        for i, vname in enumerate(variables):
            vline = '    allocate(items(%d)%%v, ' % (i+1)
            vline += 'source=V(%s, ' % (vname.lower())
            vline += 'name=trim(name)//\'%%%s\'))' % vname.upper()
            print(vline)

    @classmethod
    def template_dump_derived_type(varname, derived):
        """
        Utility method to generate boilerplate code for the Fortran
        state dump of derived types.
        """
        # Filter out private and internal attributes
        internals = ['from_handle']
        attr_names = [name for name in dir(derived)
                      if not name.startswith('_') and name not in internals]

        # Generate the boilerplate dump code
        print('    derived%%ptr => %s' % varname)
        print('    allocate(derived%%items(%d))' % len(attr_names))
        for i, aname in enumerate(attr_names):
            vline = '    allocate(derived%%items(%d)%%v, ' % (i+1)
            vline += 'source=V(%s%%%s, ' % (varname, aname.upper())
            vline += 'name=trim(name)//\'%%%s\'))' % aname.upper()
            print(vline)

    def __init__(self, filename):
        self.filepath = Path(filename)
        self.info = OrderedDict()
        self.data = OrderedDict()

        indexed = defaultdict(list)

        # Read meta information from .info file
        with self.filepath.with_suffix('.info').open() as f:
            reader = CSVReader(f, delimiter='\t')
            for row in reader:
                name = row[0]
                dtype = row[1]
                shape = tuple(int(i) for i in row[2].strip('()').split())
                index = int(row[3])

                self.info[(name, index)] = (dtype, shape)

        with self.filepath.with_suffix('.data').open() as f:
            # Read and initialdummy value (should be 1) to determine endian-ness
            should_be_one = np.fromfile(f, dtype='>i4', count=1)
            endianness = '>' if should_be_one == 1 else '<'

            for (name, index), (dtype, shape) in self.info.items():
                # Read raw data and shove into a Fortran array
                size = reduce(mul, shape)
                rtype = endianness + self._readtype[dtype]
                raw_data = np.fromfile(f, dtype=rtype, count=size)
                raw_data = raw_data.reshape(tuple(reversed(shape)))

                if size == 1:
                    data_item = raw_data[0]
                else:
                    # Force Fortran allocation in numpy before transposing into buffer
                    data_item = np.empty(shape=shape, dtype=self._alloctype[dtype], order='F')
                    data_item[...] = raw_data.transpose()

                if index >= 0:
                    # Read raw snapshot and append to list
                    indexed[name] += [data_item]
                else:
                    self.data[name] = data_item

            for name, array in indexed.items():
                self.data[name] = np.array(array)

    def recover_derived_type(self, obj, varname):
        """
        Sanitizes the raw state dump dictionary by populating a native
        composite object from the raw data and replacing the per-component
        entries. This is required to pass the derived-type object back to
        any Fortran subroutine.
        """
        keys = [key for key in self.data.keys() if '%s%%' % varname in key]
        for key in keys:
            attrkey = key[len(varname)+1:].lower()
            assert hasattr(obj, attrkey)  # Ensure attribute has been allocated
            setattr(obj, attrkey, self.data.pop(key))
        self.data[varname] = obj

    def compare(self, state):
        match = True

        for name, value in self.data.items():
            if name not in state.data:
                print('Variable not found in data2: %s' % name)
            else:
                error = value == state.data[name]
                if not np.all(error):
                    print('Data mismatch between states for variable: %s' % name)
                    match = False

        if match:
            print('State dumps match!')
        return match

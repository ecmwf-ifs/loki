# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path
import pytest

from loki import (
    HAVE_FP, HAVE_OFP, REGEX, RegexParserClass,
    FileItem,
    Sourcefile
)

pytestmark = pytest.mark.skipif(not HAVE_FP and not HAVE_OFP, reason='Fparser and OFP not available')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def test_file_item(here):
    proj = here/'sources/projBatch'

    def get_item(path, parser_classes):
        filepath = proj/path
        return FileItem(
            filepath.name.lower(),
            Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=parser_classes)
        )

    # A file with simple module that contains a single subroutine
    item = get_item('module/a_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 'a_mod.f90'
    assert item.definitions == (item.source['a_mod'],)
    items = item.get_items()
    assert len(items) == 1
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)

    # A file with a simple module that contains a single typedef
    item = get_item('module/t_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 't_mod.f90'
    assert item.definitions == (item.source['t_mod'],)

    items = item.get_items()
    assert len(items) == 1
    assert items[0].name == 't_mod'
    assert items[0].definitions == ()  # No typedefs because not selected in parser classes

    item = get_item('module/t_mod.F90', RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass)
    assert item.name == 't_mod.f90'
    assert item.definitions == (item.source['t_mod'],)

    items = item.get_items()
    assert len(items) == 1
    assert items[0].name == 't_mod'
    assert items[0].definitions == (item.source['t'],)

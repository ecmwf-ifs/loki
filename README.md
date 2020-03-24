## Loki: Freely programmable source-to-source translation

**Loki is an experimental tool(!)** to explore the possible use of
source-to-source translation for the IFS and ESCAPE dwarfs. Loki is
based on compiler technology (visitor patterns and ASTs) and aims to
provide an abstract, language-agnostic representation of a kernel, as
well as a programmable (pythonic) interface that allows developers to
experiment with different kernel implementations and
optimizations. The aim is to allow changes to programming models and
coding styles to be encoded and automated instead of hand-applying
them, enabling advanced experimentation with large kernels as well as bulk
processing large numbers of source files to evaluate different kernel
implementations and programming models.

### Installation

See [INSTALL.md](INSTALL.md).

### Core concepts (the philosophical bit)

Please note that, on a fundamental level, converting between different
styles in a low-level compiled language like Fortran or C/C++ often
requires assumptions to be made that are model-specific and do not
generalize to the entire language. This is why Loki provides a
programmable interface rather than a push-button solution, leaving it
up to developers to decide which assumptions about the original source
code can be used and how. For example, converting large numbers of IFS
physics code to a "single column" format (see below) requires the
explicit knowledge of which index variables typically represent the
parallel dependency-free horizontal dimension that is to be lifted.

The aim of Loki is therefore to give developers all the tools to encode their
own code transformation in an elegant, pythonic fashion. The core concepts
provided for this are:

* `Module` and `Subroutine` classes (kernels) that each provide an
  _Intermediate Representation_ (IR) of their source code, as well as
  utilities to inspect and transform the underlying IR nodes.
* Expressions contained in `Statement`, `Loop` and `Conditional` nodes
  are represented as independent sub-trees, based on the
  [Pymbolic](https://github.com/inducer/pymbolic) infrastructure.
* Three frontends are supported that are used to parse Fortran code
  either from source files or strings into the Loki IR trees. Two
  backends are provided to generate Fortran or (experimentally) C code
  from the combined IR and expression trees.
* A `Transformation` class is provided that allows users to encode
  individual code changes based on the abstract representation
  provided by Loki's IR and expression objects and can be applied
  to individual `Subroutine` and `Module` objects - much like simple
  compiler passes).
* A `Scheduler` class (work in progress) that provides bulk processing
  and inter-procedural analysis (IPA) tools to apply individual change
  over large numbers of files while honoring the call-tree that
  connects them.

### Example transformations and current features

Loki is primarily an API and toolbox, requiring developers to create their
own head scripts to create and invoke source-to-source translation toolchains.
The primary set of example transformations lives under `scripts/transform.py`,
and the `loki_transform.py` script is provided by the Loki install. The primary
transformation passes provided are:

* **Idempotence (Idem)** - A simple transformation that performs a
    neutral parse-unparse cycle on a kernel.

* **Single column abstraction (SCA)** - Transforms a set of kernels
  into Single column format by removing the specified horizontal
  iteration dimension. This transformation has a "driver" and a
  "kernel" mode, as it potentially changes the subroutines call
  signature to remove derived types (structs do not expose
  dimensions).

* **RAPS integration** - A special set of tools is provided that can
  bulk-transform large numbers of source files while honoring
  dependencies due to Fortran module imports
  (`scripts/scheduler`). This has been integrated with RAPS, which
  requires an additional step to inject a modified "root" source file
  into an IFS build from which a modified dependency tree (for example
  in SCA format) is invoked.

* **C transpilation** - A dedicated Fortran-to-C transpilation
  pipeline that converts Fortran source code into (column major,
  1-indexed) C kernel code. The transformation pipeline also creates
  the necessary header and `ISOC` wrappers to integrate this C kernel
  with a Fortran driver layer, as demonstrated with the CLOUDSC ESCAPE
  dwarf.

### Quick start and basic usage

#### Parsing and Frontends

To read and parser existing source files Loki provides multiple frontends:
```
source = SourceFile.from_file(path, ..., frontend=OFP|OMNI|FP)
routine = source['my_subroutine']
```
Alternatively, for testing purposes, we also allow direct construction of
`Subroutines` from source code strings:
```
fcode = """
  subroutine test_routine
    write(*,*) 'Hello world!'
  end subroutine test_routine
"""
routine = Subroutine.from_source(fcode, frontend=OFP|OMNI|FP)
```

Due to the inherently convoluted nature of the Fortran language, no
freely available frontend is feature complete. For this reason we
currently support three frontends, each with known limitations:

* [FP - FParser](https://github.com/stfc/fparser) - A "pure Python"
  frontend developed by STFC. Was added last, but is increasingly
  becoming the first choice frontend, due to easy integration (no
  Java) and active development and support (thanks Rupert!). Will
  likely become the default once all necessary features required for
  our demonstators are safely integrated and tested (soon!).

* [OMNI](http://omni-compiler.org) - Part of the XOpenMP
  framework. Performs parser-specific pre-processing and inlines
  external includes, which requires dependency chasing in large code
  bases. Some other caveats: It inserts explicit lower bounds of `1`
  into array size declarations and lower-cases everything!

* [OFP](https://github.com/mbdevpl/open-fortran-parser-xml) - A
  Python-wrapper by [mbdevpl](https://github.com/mbdevpl) around the
  [Open Fortran
  Parser](https://github.com/OpenFortranProject/open-fortran-parser)
  (ROSE frontend). A little brittle in places and requires a few ugly
  workarounds, but it does provide line numbers and is very good for
  preserving the "look-and-feel" of original code.

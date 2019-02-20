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

Loki is pure Python package that depends on a range of upstream packages,
including some dependencies on dev branches. It is therefore recommended
to create a Loki-specific virtual environment:

```
# Create virtual env
python3 -m venv loki_env
source loki_env/bin/activate
pip install --upgrade pip
pip install numpy

# Clone and install dev version of Loki
git clone ssh://git@git.ecmwf.int:7999/~naml/loki.git
cd loki
pip install numpy  # Needed during next step
pip install -r requirements.txt
pip install -e .  # Installs Loki dev copy in editable mode
```

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

* ``Module` and ``Subroutine`` classes (kernels) that each provide an
  abstract syntax tree (AST) of their source code, as well as utilities.
* Assignments, loops and conditional each contain expressions of the actual
  computation to perform with a set of variables. These expressions consist
  of extended [SymPy](https://sympy.org) expressions, which provide powerful symbolic
  capabilities (more below).
* ``Transformation`` class, as well as bulk processing and inter-procedural
  analysis (IPA) tools that allow bulk processing large numbers of files
  while honoring the call-tree that connects them. `Transformation`s are
  customizable and are intended as the primary tool to encode a specific
  set of changes (much like compiler passes).

### Examples and current features

Loki is primarily an API and toolbox, requiring developers to create their
own head scripts to create and invoke source-to-source translation toolchains.
The primary set of example transformations lives under `scripts/transform.py`.

#### Idempotence (Idem)

A simple transformation that performs a neutral parse-unparse cycle on a kernel.

#### Single column abstraction (SCA)

Transforms a set of kernels into Single column format by removing the
specified horizontal iteration dimension. This transformation has a "driver"
and a "kernel" mode, as it potentially changes the subroutines call signature
to remove derived types (structs do not expose dimensions).

#### RAPS integration

A special set of tools is provided that can bulk-transform large
numbers of source files while honoring dependencies due to Fortran
module imports (`scripts/scheduler`). This has been integrated with RAPS,
which requires an additional step to inject a modified "root" source file
into an IFS build from which a modified dependency tree (for example in
SCA format) is invoked.

#### C transpilation

A dedicated Fortran-to-C transpilation pipeline that converts Fortran source
code into (column major, 1-indexed) C kernel code. The transformation pipeline
also creates the necessary header and `ISOC` wrappers to integrate this C kernel
with a Fortran driver layer, as demonstrated with the CLOUDSC ESCAPE dwarf.

### Further development considerations

#### Frontends

To read and parser existing source files Loki provides multiple frontends:
```
source = SourceFile.from_file(path, ..., frontend=OFP/OMNI)
routine = source['my_subroutine']
```

We currently support two frontends:

* [OFP](https://github.com/mbdevpl/open-fortran-parser-xml) - A
  Python-wrapper by [mbdevpl](https://github.com/mbdevpl) around the
  [Open Fortran
  Parser](https://github.com/OpenFortranProject/open-fortran-parser)
  (ROSE frontend). A little brittle in places and requires a few of
  ugly workarounds, but it does provide line numbers and is best for
  preserving the "look-and-feel" of original code.

* [OMNI](http://omni-compiler.org) - Part of the XOpenMP
  framework. Performs parser-specific pre-processing and inlines
  external includes, which requires dependency chasing in large code
  bases. Some other caveats: It inserts explicit lower bounds of `1`
  into array size declarations and lower-cases everything!

#### SymPy integration

Expressions (variables, calculations and logical conditions) are
[SymPy](https://sympy.org) objects, which means that we can
programmatically detect symbolic equivalence (eg. `2*(a + b) == 2*a +
2*b`). This is a powerful tool that is useful for various types of
advanced code analysis source manipulation, but it does introduce some
complexity with regards to symbol caching and scoping (see
`loki.expression.symbol_types`).

In addition, recent versions of SymPy provide a range of code
generation capabilities to nearly all relevant programming languages,
which means we can utilize its `CodePrinter` API to create
neat-looking source code in our backends.  This again comes with some
caveats - SymPy's expression rewriter can easily introduce round-off
error. To prevent this we use SymPy's `evaluate(False)` utility in
various places, and care needs to be taken during expression
replacement.

#### API caveats due to SymPy (TL;DR) :
* Source code variables are either of the type `Scalar` (a
  `sympy.Symbol`) or `Array` (a `sympy.Function`). Both can be created
  via the universal `Variable(name='f', dimensions=(x, y))`
  constructor.

* Variable instances are cached, either globally if created via
  `Variable(name='f')` or on a kernel-specific context if created
  via `routine.Variable(name='f')`. This caching means that meta-data,
  like the data type of a variable or the shape of an `Array` are shared
  between all instances (occurrences) of a variable within a kernel context.

* Like all symbols in SymPy, variable instances are not meant to be
  changed, but re-generated and substituted into expressions. To preserve
  our context-sensitive caching we therefore provide a `variable.clone(...)`
  method that re-generates all components of a variable that have not been
  explicitly provided and re-generates the modified symbol in the same
  caching context. A worked example, which removes the first dimension
  of all arrays if it is the desired target dimensions:
  ```
  vmap = {}
  target = <target dimension to replace>
  for v in FindVariables().visit(routine.body):
      if v.is_Array and v.dimensions[0] == target:
          vmap[v] = v.clone(dimensions=v.dimensions[1:])
  routine.body = SubstituteExpressions(vmap).visit(routine.body)
  ```

* The treatment of boolean logic in SymPy is fundamentally different
  to numbers, so a variable can either be a boolean or a number. To get
  around this, we have hidden special types for boolean variables. But(!)
  for this trick to work we need to know the type on variable instantiation,
  making it necessary for the `Subroutine` constructors to store the types
  of local variables and pass them to the frontend when parsing the
  routine body.

#### Wishlist (more TL;DR):

* "Smart" configuration dict that provides callbacks, understand env
  variables and provides different codegen styles
* Integration with the STFC `fparser` frontend (natively Python)
* Closer integration of native SymPy types, expecially for data type handling
* Introduce conceptual "dimensions" as a first-class concept; make handling
  of array sizes and indices/index ranges more explicit and elegant.
#!/usr/bin/env python3 

import os, sys, getopt, string,argparse,re,subprocess,copy,shutil,time
import click as cli

import ecir.helpers as helpers

print_routine_names = True

def usage():
    print( \
"""
add self-explanatory usage message here!
""" )

@cli.command()
@cli.option('--file', '-f', multiple=True,
            help="Explicitly defined source file(s).")
@cli.option('--projectfile', '-p', default=None,
            help="Project file containing multiple source file names.")
def main(projectfile, file=None):
    t0 = time.time()

    ## Initialization...

    lines = 0           # count parsed lines
    f=0                 # file counter
    warning=0           # if duplicate modules are found, warning=1
    verbose = True      # print stuff to stdout


    # list of files that, for whatever reason, should not be scanned
    do_not_scan_list = ['Magics_dummy.F90'] 

    # Gnerate source file list either from project file or given file list
    source_files = list(file)
    if projectfile is not None:
        # Process project file for long file lists
        print("Reading project file: %s" % projectfile)
        with open(projectfile) as f:
            source_files += [line.strip() for line in f.readlines()]
    print("Source files: %s" % str(source_files))

    ### Parse the project file and reload the current dependency map from .spam_restart_file

    #print project_file
    helpers.classify_files(source_files)

    t1 = time.time()
    print('classify_files elapsed time : ', t1-t0)

    ### Next, generate a dictionary of routines provided by modules.

    print("\nCaching source code for all routines to be scanned :   ")
    for (filename,(type, is_include, compflag, mtime, deps, reparse, deleted)) in sorted(helpers.clean_source_files2.items()):

        helpers.store_source_code(filename)


    t2 = time.time()
    print('store_source_code elapsed time : ', t2-t1)


    ### First pass. look up definitions of e.g. modules and subprograms.

    print("\nRegistering standalone routines, and finding multiple standalone routines in a single file :   ")


    ignore_these_multiple_routine_files = ['ifsaux','fp_serv_suiosctmpl.F90']

    for (filename,(type, is_include, compflag, mtime, deps, reparse, deleted)) in sorted(helpers.clean_source_files2.items()):

        #print('scanning ',filename)
        if any( wd in filename for wd in do_not_scan_list ) :
            continue
        if not reparse:
            continue
        # scan line by line for modules, routines, program etc
        #helpers.prelim_scan_file(filename,type)
        helpers.scan_multiple_standalones_per_file(filename,type,ignore_these_multiple_routine_files)



    t3 = time.time()
    print('first scan of files elapsed time : ', t3-t2)
    print("")


    t4 = time.time()
    #print('register_standalone_routines elapsed time : ', t4-t3)
    print('A total of ',len(helpers.standalone_routines.keys()),' standalone routines found')

    do_not_check_interfaces = ['ifsaux','tm5_kpp_Integrator.F90']

    for (filename,(type, is_include, compflag, mtime, deps, reparse, deleted)) in sorted(helpers.clean_source_files2.items()):

        if any(routine in filename for routine in do_not_check_interfaces) :
            continue

        # scan files to identify which routines need an interface to be called
        ignore_these_calls = ['GSTATS','GSTATS_BARRIER','GSTATS_BARRIER2','ABOR1']
        helpers.check_for_includes(filename,type,ignore_these_calls)

    t5 = time.time()
    print('check_for_includes elapsed time : ', t5-t4)
    print("Finished scanning files for missing interfaces \n ")

    for (filename,(type, is_incl, compflag, mtime, deps, reparse, deltd)) in sorted(helpers.clean_source_files2.items()):

        # scan files for preparatory CLAW work
        decl_lines = helpers.register_routine_source(filename)

        #helpers.count_nested_loops_depth(filename)

        # get list of arrays needing to have horiz dim removed
        arrs_to_demote = helpers.identify_horiz_loop_arrays(filename)

        # see if we can findall these arrays in declarations
        helpers.find_arrays_in_declarations(decl_lines, arrs_to_demote, filename)

        #try to generate CLAW-compatible code
        helpers.generate_single_column_code(filename, arrs_to_demote)

        # CLAW / OMNI doesn't like associate statements, so inline the associations
        helpers.remove_associates(filename)

    #============================================================================


    if print_routine_names  and len(helpers.problem_routines.keys())>0 : 
        print("IFS files with missing interfaces were found :\n")
        for rout_name in helpers.problem_routines.keys():
            for call_name in  helpers.problem_routines[rout_name]:
                print(rout_name+' should have an interface for call to '+call_name)
        print("==============================================\n")


    # return zero for success, >0 otherwise 
    #return len(helpers.problem_routines.keys())
    if len(helpers.problem_routines.keys()) == 0:
        sys.exit(0)
    else :
        sys.exit(1)
    
if __name__ == "__main__":
    main()


import sys, string, os, re, subprocess,copy,shutil,math,time


clean_source_files2={}         
source_code_dict = {}
# a dictionary of {standalone name, [calling locations]}
standalone_routines = {}
# dictionary of {routines with problems, [problem calls]}
problem_routines = {}
# a dictionary to record whether routine name is used multiple times or not
multi_named_routines = {}
routines={}         
functions={}        
programs={}




# matcher for a valid and upper case Fortran name
match_fortran_name = re.compile("[A-Z][A-Z0-9_]*")  ## any valid name for variable or routine in Fortran code
match_single_quote = re.compile("'")
match_double_quote = re.compile('"')
match_any_quote    = re.compile('["\']')
match_exclam       = re.compile("!")
up_to_ampersand   = re.compile('(.*?)&')
whole_cont_line    = re.compile('( *&)?(.*)')


decl_keywords = ['INTEGER', 'REAL', 'DOUBLE', 'LOGICAL', 'TYPE', 'CHARACTER', 'EXTERNAL', 'COMPLEX', 'INTRINSIC', 'PARAMETER', 'DATA', 
                  'NAMELIST'] # should 'SAVE' be in this list? not obvious for the moment


def classify_files(source_files):
    """
    go through the list of file names, assign default file types by suffix,
    look for duplicate files and include directories.
    """

    ## the default guesses for file types
    
    filetype = { ".F":"fixed", ".F90":"free", ".f":"fixed", \
                 ".f90":"free", ".c":"C", ".h":"C", ".H":"C",\
                 ".inc":"fixed", ".INC":"fixed", ".SYM":"fixed"}

    longnamesize=20
    shortnamesize=12
    shortnamelist={}
    dups=0

    line_number=0

    for line in source_files:

        if any(unwanted in line for unwanted in ['fft992.F']) : 
            continue

        line_number+=1
        current_line = line.strip()


        ## if the previous parsing was without parsing, force a full reparse
        reparse=0

       
        ## keep comments at end of line
        comment=""
        cpos=current_line.find("#")
        if cpos>-1:
            comment=current_line[cpos:]
            current_line=current_line[:cpos]

        current_line = current_line.split()
        cols=len(current_line)

        longname = current_line[0]

        try:
            statobj = os.stat(longname)
        except OSError:
            print( "spam error: please check your file list, could not find file:",longname)
            sys.exit()
        
        if longname in clean_source_files2 : ## clean_source_files2.has_key(longname):
            # source file is modified after last parse
            if statobj.st_mtime > clean_source_files2[longname][3]:
                reparse = 1
                
        shortname = longname.split("/")[-1].strip()
        dotpos = shortname.rfind(".")
        suffix = shortname[dotpos:]


        ## the default group for object files
        group_name="OPT"
        if cols > 1 :
            group_name=current_line[1]
        do_opt=group_name

        if cols > 2 :
            type=current_line[2]
            if type not in ['free','fixed','fixed132','C']:
                print( 'line',line_number,'column 3 "'+type+\
                       '" should preferrably be one of fixed,fixed132,C or free:')
        else:
            if suffix in filetype: ## filetype.has_key(suffix):
                type=filetype[suffix]
            else:
                type="unknown"

        # if the file type has changed, we may need to parse in a different way
        if longname in clean_source_files2 : ## clean_source_files2.has_key(longname):
            if type != clean_source_files2[longname][0]:
                reparse=1
            
                
        inc="source"
        is_include=0
        if cols>3:
            inc=current_line[3]
            if inc not in ['include','source']:
                print( 'line',line_number,'column 4 "'+inc+'" must be one of include,source:')


            if inc=="include": is_include=1
        else:
            if suffix.lower() in [".h",".i",".inc",".com",".sym"]:
                is_include=1
                inc="include"
                

            
        # if is_include has changed, and it has changed to 0, reparse
        if longname in clean_source_files2 : ## clean_source_files2.has_key(longname):
            if is_include != clean_source_files2[longname][1] and not is_include:
                reparse=1
                
        slashpos = longname.rfind("/")
        dirname = longname[:slashpos]



        if cols>4:
            themodifiedline = line
            fullmodifiedline = line
        else:
            if len(longname) >= longnamesize: longnamesize = len(longname) + 2
            themodifiedline = longname.ljust(longnamesize) + \
                              group_name.ljust(10) + \
                              comment + "\n"
            fullmodifiedline = longname.ljust(longnamesize) + \
                              group_name.ljust(10) + \
                              type.ljust(9) + \
                              inc.ljust(8) + \
                              comment + "\n"

        if reparse or not longname in clean_source_files2 : ## clean_source_files2.has_key(longname):
            deps = []
            reparse = 1
        else:
            deps = clean_source_files2[longname][4]
            
        statobj=os.stat(longname)

        # a file that will be processed, remove the default deleted status:
        deleted=0
        
        clean_source_files2[longname] = (type, is_include, do_opt, statobj.st_mtime, deps, reparse, deleted)
        
        if shortname in shortnamelist: ## shortnamelist.has_key(shortname):
            shortnamelist[shortname].append(longname)
            dups=1
        else:
            shortnamelist[shortname]=[longname]

    
    ## delete files not in project file
    for key in clean_source_files2.keys():
        if clean_source_files2[key][6]:
            print( "INFO:", key, " has apparently been removed from the list of source files")
            del clean_source_files2[key]

                
            # also remove object file from dependency lists
            for key2 in clean_source_files2.keys():
                dotpos = key.rfind(".")
                objf = key[:dotpos] + ".o"
                #print "obj to remove=",objf, "current key=", key2, "deps=", clean_source_files2[key2][4]
                try:
                    clean_source_files2[key2][4].remove(objf)
                except ValueError:
                    pass
                       


    if dups:
        print( "\nIn the given list of source files, some of the file names were")
        print( "identical, which indicates a duplicate. See file .ifsanalysis_duplicates")
        print( "for a list over files with identical names.\n")
        snf = open('.ifsanalysis_duplicates','w')
        for key in shortnamelist:
            if len(shortnamelist[key])>1:
                if len(key)>shortnamesize: 
                    shortnamesize = len(key)+2
                ## snf.write(key+": " + string.join(shortnamelist[key]) + "\n")  ## deprecated
                snf.write(key+": " + ' '.join(shortnamelist[key]) + "\n")
        snf.close()


#    return includepath


#======================================================

def store_source_code(filename) :
    try :
        out = (open(filename,'rt', encoding="ascii", errors="surrogateescape")).read()
    except IOError :
        print('is file ',filename,' missing ?')
    
    # store the naked list of text lines from the file in a dictionary hashed with file name
    source_code_dict[filename] = [l+'\n' for l in out.split('\n')]

#======================================================

def register_routine_source(filename) :

    src_lines = source_code_dict[filename] # not yet a tuple, we hope!
    if isinstance(src_lines,tuple) : # means this method has been called more than once on this file?
        src_lines = src_lines[0]
    src_iter = iter(src_lines)

    get_all_src = False
    #get_all_src = True

    position_in_file = 'before_routine'
    long_line = ''
    routine_code_lines = []
    routine_decl_lines = []
    
    # identify main routine being defined in this file
    routine_name = ''
    for line in src_lines : 
        if line.upper().lstrip().startswith('SUBROUTINE') :
            wds = re.split('\W+', line.upper()) 
            if len(wds) == 1 : 
                raise SystemError('how can this be happening?? '+filename)
            routine_name = wds[1]
            print('Identified ',routine_name,' in ',filename)
            break
    
    if get_all_src:  # we want everything between 'subroutine xxx' and 'end subroutine xxx'
        
        ## read in joined line by joined line and attribute lines to the right variable
        for line in (src_iter):
            ## join lines that should be joined
            #long_line,orig_lines = assemble_continued_statement(line, src_lines)
            long_line,orig_lines = assemble_continued_statement_from_iterator(line, src_iter)
            
            ## split joined line into words
            words = re.split('\W+',long_line.lstrip().upper())
            
            ## have we found the routine?
            if not long_line.lstrip().startswith('!' ) and \
               (words[0] in ['SUBROUTINE','FUNCTION'] and words[1] == routine_name  or \
                (len(words)>2 and all( el in words for el in ['FUNCTION' , routine_name]) and words[0] != 'END' ) ):
                print('Found routine in file!')
                position_in_file = 'in_routine'
                
                routine_code_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                long_line = ''
                continue
        
            
            if position_in_file=='in_routine' and words[0]=='END' and words[1] in ['SUBROUTINE','FUNCTION'] and \
                (len(words)==2 or words[2]==routine_name):
                routine_code_lines.append( (long_line, orig_lines) )
                position_in_file = 'after_routine'
                long_line = ''
                continue
            
            
            if position_in_file == 'in_routine':
                routine_code_lines.append( (long_line, orig_lines) )

            long_line = ''



    else:  # we don't take all the code '
        ## read in joined line by joined line and attribute lines to the right variable
        for line in src_iter:
            ## join lines that should be joined
            #long_line,orig_lines = assemble_continued_statement(line, src_lines)
            long_line,orig_lines = assemble_continued_statement_from_iterator(line, src_iter)
       
            ## split (unjoined) line into words
            words = re.split('\W+',long_line.lstrip().upper())
            
            ## have we found the routine?
            if not long_line.lstrip().startswith('!' ) \
               and (words[0] in ['SUBROUTINE','FUNCTION'] and words[1] == routine_name or 
                    (len(words)>2 and all(el in words for el in ['FUNCTION', routine_name]) ) ):
                print('Found routine in file!')
                position_in_file = 'in_comments'
                long_line = ''
                continue
            
            ## deal with the comments zone
            if position_in_file=='in_comments':
                if long_line.lstrip()=='' :
                    long_line = ''
                    continue
                elif long_line.lstrip()[0]=='!':
                    #print('comment line : ',long_line)
                    long_line = ''                    
                    continue
                else:  ## we've got to the end of the comments 
                    position_in_file = 'in_MODULES'
                    #routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                    #long_line = ''
                    #continue
    
        
            ## deal with the USE MODULES
            if position_in_file=='in_MODULES':
                if words[0]=='USE' or line.lstrip()=='' :
                    #routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                    #print('USE line : ',long_line)
                    long_line=''
                    continue
                elif long_line.lstrip().startswith('!') : # dont really care about this line for the moment
                    routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                    print('USE line : ',long_line)
                    long_line=''
                    continue
                elif words[0] == 'IMPLICIT' : # use this to switch explicitly to 'in_declarations'
                    position_in_file = 'in_declarations'
                    #routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                    print('DECL line : ',long_line)
                    long_line=''
                    continue

                else:  ## we've got to the argument declarations
                    #routine_argument_specs_list.append(long_line)
                    position_in_file = 'in_declarations'
                    routine_decl_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                    long_line = ''
                    continue
        
        
            ## deal with the variable declarations, header includes, and ASSOCIATE
            if position_in_file=='in_declarations':
                if words[0]=='IF' and 'LHOOK' in words: ## this is the end of declarations
                    #line=''
                    position_in_file = 'routine_ASSOCIATE'
                    long_line = ''
                    continue
                elif long_line.startswith('#include') or long_line.lstrip().startswith('!') or long_line.strip()=='':
                    #routine_decl_lines.append( (long_line, orig_lines) ) # dont actually need to store this line
                    long_line = ''
                    continue                    
                elif not long_line.lstrip().startswith('!') and words[0] not in decl_keywords : 
                    print("we don't seem to have a DrHook call for this routine : ", filename, '*'+long_line+'*')
                    input('...')
                    position_in_file = 'routine_ASSOCIATE'
                    long_line = ''
                    continue
                elif line.strip() == '' : 
                    #routine_decl_lines.append( (long_line, orig_lines) ) dont need to store
                    long_line = ''
                    continue
                else:  ## we're still in the argument declarations
                    #routine_decl_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                    routine_decl_lines.append( (strip_comments(long_line), orig_lines) ) ## added to check declarations for arg intents
                    long_line = ''
                    continue

            ## deal with ASSOCIATE
            if position_in_file=='routine_ASSOCIATE':
                if words[0] != 'ASSOCIATE':
                    print('We seem to have no ASSOCIATE for '+routine_name+'\n')
                    
                    position_in_file = 'in_routine'
                    routine_code_lines.append( (long_line, orig_lines) )
                    long_line = ''
                    continue
                else:
                    
                    routine_code_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                    position_in_file = 'in_routine'
                    long_line = ''
                    continue



            if position_in_file=='in_routine' and words[0]=='END' and words[1] in ['SUBROUTINE','FUNCTION'] and \
               (len(words)==2 or words[2]==routine_name):
                routine_code_lines.append( line )
                position_in_file = 'after_routine'
                long_line = ''
                continue
            
            
            if position_in_file == 'in_routine':
                routine_code_lines.append( (long_line, orig_lines) )
    
            long_line = ''

    if False : 
        print('Declaration lines for ',routine_name,' : \n')
        print ( [line[0] for line in routine_decl_lines] )
        print('=============================')

    temp = source_code_dict[filename]
    if not isinstance(temp, tuple): # not a tuple, so haven't run this yet on this file
        source_code_dict[filename] = (temp, routine_name) # store name of first routine defined in file
    print('regist_source returning tuple of length ',len(routine_decl_lines))
    return routine_decl_lines
#end def register_routine_source(filename)
        
#======================================================

def register_standalone_routines(filename) :
    
    in_module = False
    standalone_name = ''
    in_routine = 0

    pr = False
    #if 'suafn3.F90' in filename: pr = True
    if pr : print('debugging register_standalone_routines for ',filename)

    src_lines = source_code_dict[filename][0]
    line_num = 0 # new V2
    src_iter = iter(src_lines)
    
    for line_num,line in enumerate(src_iter) : # new V2

        l_l = assemble_continued_statement_from_iterator(line, src_iter,False)

        if l_l.startswith('!') : continue # no point in doing unnecessary checking !
        if l_l.startswith('#') : continue # no point in doing unnecessary checking !
        
        words = re.split( '\W+', l_l.strip().split('!')[0].upper() )


        if len(words)==2 and words[0] == 'MODULE' and in_module == False : # module definition
            in_module = True
            module_name = words[1]
            if pr: print('register_standalone found module :',module_name)
            break # MODULE keyword has been found before subroutine / function 

        elif len(words) >= 3 and words[0] == 'SUBROUTINE' : ## >=3 rather than 2, to avoid registering routines that have no argument
            standalone_name = words[1]
            if pr: print('register_standalone found subr :',standalone_name)
            break # we have found a standalone subroutine
        elif len(words) >= 4 and words[1] == 'SUBROUTINE' and words[0] in ['ELEMENTAL','PURE'] :  ## >=4 rather than 3, see above
            standalone_name = words[2]
            if pr: print('register_standalone found subr :',standalone_name)
            break # we have found a standalone subroutine
        elif len(words) >= 2 and 'FUNCTION' in words and words[0] != 'END' : 
            standalone_name =  words[ words.index('FUNCTION')+1]
            if pr: print('register_standalone found func :',standalone_name)
            break # we have found a standalone function
        elif len(words) >= 2 and words[0] == 'PROGRAM' : 
            standalone_name =  ''
            if pr: print('register_standalone found program :',words[1])
            break # we have found a standalone program
            

    #fle.close()
    # stick the information in a standalone_routines dictionary
    if standalone_name != '':
        standalone_routines[standalone_name] = []

#end register_standalone_routines

#======================================================

# scan line by line for modules, routines, program etc
def check_for_includes(filename,type,ignore_missing_interfaces):

    srclines = copy.copy(source_code_dict[filename])

    long_line = ""      # the line that will be sent to the splitter
    stored_line = ""    # the statement following long_line
    last_line=0         # if true, fline is the last line of a file.
    current_routine = ''# self explanatory, I think
    current_module  = ''# self explanatory, I think
    containing_routine = '' # only defined if we find a routine containing another ...
    includes_list = []  # store the list of interfaces pulled in by #include, for each routine / function
    module_includes = [] # any includes in a module before the routines get stored in this and prepended to includes_list

    # variables to deal with prepending routine names
    in_module_bool   = False
    in_routine       = 0
    in_contains_bool = False
    in_explicit_interface = False
    parent_name = ''

    pr = False
    #if 'larche2.F90' in filename : pr = True # switch on debugging printing for particular file

    iter_lines = iter(srclines)

    list_of_concat_lines = [] # to be passed as argument to subroutine match_call, to deal with type-bound procedures 

    for line_num,fline in enumerate(srclines):

        ## the scanner. feed each line into the joiners

        if type=="free":
            long_line, get_next = join_and_strip_free(fline, long_line)

        elif type in ["fixed","fixed132"]:
            last_line = cline==len(srclines)

            long_line, get_next, stored_line, warning_flag = join_and_strip_fixed(\
                fline, long_line, stored_line, last_line, type=type)

            if warning_flag & 2**0: file_tabs.append(cline)
            if warning_flag & 2**1: cont_cpp.append(cline)
        else:
            long_line, get_next = fline.strip(), 0

        if get_next:
            continue ## break out, back to loop for with next fline

        # commented out strip_strings here because it breaks interface include detection
        #long_line = strip_strings(long_line)

        if list_of_concat_lines == [] : 
            list_of_concat_lines = [long_line]
        else : 
            list_of_concat_lines.append( long_line )

        ## the splitter. split each line at ;
        ## this gives us one statement per line
        for line in long_line.split("$$$$$"):
            line = line.strip()
        #if True:
        #    line = copy.copy(long_line).strip()

            if line == "":
                continue
            elif line.startswith('!') :
                continue
            #lines+=1

            #line = helpers.strip_strings(line)
            line = line.upper()  ## return line in UPPERCASE characters

            fortran_names = match_fortran_name.findall(line)
            #if pr: print('working on line : \n',line)

            # deal with preprocessor lines
            ##if line.startswith('#IF'):
            ##elif line.startswith('#EL') or line.startswith('#END'):

            if line.strip() == 'INTERFACE' :
                in_explicit_interface = True
            elif len(fortran_names) ==2 :
                if fortran_names[0] == 'END' and fortran_names[1] == 'INTERFACE':
                    in_explicit_interface = False

            #  first, check if we are in a module or not!
            name = match_module(fortran_names) ## return name of module if line contains a declaration, or empty string otherwise             
            if name != "":   ## we have a module declaration on this line
                in_module_bool = True
                current_module = name
                parent_name = re.split('\W+',line)[1]  + '%'
                if pr: print('found a module: ',line,parent_name,line_num, '\n')
                continue  # if we have found a module, no need to look for anything else on the same line!

            ## look for end of module
            name = match_module_end(fortran_names)
            if name != "":
                current_host=""
                contained=0

                if in_module_bool == True: 
                    in_module_bool = False
                    in_contains_bool = False
                    parent_name = ''
                    if pr: print('found end of  module: ',line,parent_name,line_num, '\n')
                else: # we seem to have found an END MODULE with no corresponding start ...
                    raise SystemError('check '+filename+' for END MODULE with no MODULE to start it ...')
                continue  # if we have found end module, no need to look for anything else on the same line!


           

            # fill in includes list against which to check routine calls ## WHERE SHOULD THIS LIST BE RESET? ENTERING ROUTINE?
            include_name = match_include(line,fortran_names)
            if include_name != '' :
                if current_routine == '' :
                    if pr : print('case of interface included out of routine? check ',filename)
                    #input(' interface in question : '+include_name )
                    module_includes.append( include_name )
                else :
                    includes_list.append(include_name)
                if pr : print('found an #include in ', current_routine,' : ',include_name)
                
            ## detect SUBROUTINEs, FUNCTIONs
            declaration_line = 0
            ## detect SUBROUTINE definition
            name = match_subroutine_def(fortran_names)
            if name != "" :
                if len(list_of_concat_lines) > 2: 
                    if list_of_concat_lines[-2].strip() == 'INTERFACE':
                        in_explicit_interface = True
                in_routine += 1
                if pr : print ( 'just incremented in_routine for sbrt: ', in_routine, line , ' --- ',name, '---',fortran_names)
                if current_routine != '' : # we are already in a routine!
                    containing_routine = current_routine
                if pr : print('changing current_routine from ',current_routine,' to ',name)
                current_routine=name
                # reset includes_list for this routine, unless it is a contained routine
                if containing_routine == '':
                    includes_list = []
                if module_includes != []:
                    includes_list = copy.copy(module_includes)


                #current_routine = parent_name + name
                if pr: print('found subrout def: ',line,parent_name,current_routine,line_num, '\n')

                continue # no need to go on

            ## detect FUNCTION
            try :
                name = match_function_def(fortran_names,line)
            except:
                print('origin of failure :',line,filename)
                sys.exit()
            if name !="" :
                in_routine += 1
                if pr : print ( 'just incremented in_routine for fct : ', in_routine )
                if in_routine > 1: 
                    in_contains_bool = True
                if pr : print('changing current_routine from ',current_routine,' to ',name)
                current_routine=name
                
                # reset includes_list for this function
                includes_list = []
                if module_includes != []:
                    includes_list = copy.copy(module_includes)

                declaration_line =1          
                if pr: print('found function def: ',line,parent_name,line_num, '\n')
                continue # no need to go on

            ## detect PROGRAM definition
            name = match_program_def(fortran_names)
            if name!="":
                in_routine += 1
                if pr : print ( 'just incremented in_routine for prg: ', in_routine )
                if pr : print('changing current_routine from ',current_routine,' to ',name)
                current_routine=name
                current_routine_fullname = name
                declaration_line =1
                continue # no need to go on


            ## detect end of (SUBROUTINE/FUNCTION/PROGRAM)
            if match_end_subprogram(fortran_names):  # returns true if list is not empty
                if containing_routine == '':
                    current_routine=""
                else :
                    current_routine = containing_routine
                    containing_routine = ''
                in_routine -= 1
                if in_explicit_interface :
                    in_explicit_interface = False
                contained=0
                declaration_line =1                    
                if pr: print('found end of subrout: ',line,line_num, '\n')
                continue # no need to go on

            # detect CONTAINS statement 
            if line.strip().upper() == 'CONTAINS':
                in_contains = True
                continue # no need to go on


            ## detect CALL SUBROUTINE
            name = match_subroutine_call_simple(fortran_names,line,list_of_concat_lines,filename)
            if name!="":
                if current_routine=="":
                    #if 'DR_HOOK' in line:  # this is a case I haven't properly parsed, but safe to ignore ...                                
                    #    print('')
                    #else:  # subroutines containing explicit interface definitions can do this
                    #    print( "\nmost likely not in a routine, but nevertheless found calls to subroutine ",name)
                    #    print( "check manually is going on")
                    #    print( "file=",filename,"line=",line)
                    continue

                else:   # we have a name for the routine called, now check if it's the one we used at first scan 
                    if pr: 
                        print( ' found a routine call : ',line,'...',name,' for ',current_routine,name in standalone_routines.keys())
                    if name in standalone_routines.keys() and name not in includes_list and name not in ignore_missing_interfaces:
                        if pr : print('this has no interface! ', name)
                        # this is a call without an interface !                        
                        if current_routine in problem_routines.keys() :
                            if name not in problem_routines[current_routine] :
                                problem_routines[current_routine].append(name)
                        else :
                            problem_routines[current_routine] = [name]
                        #print('we found a problem in ',filename, ' ... ',name,' --- ', line)
                        #print('includes_list : ', includes_list )
                        #print('  ')
                        #input('check call to '+name)

                continue # no need to go on


                
            # look for function calls on all lines except beginning or end of definition
            if not declaration_line:
                functions_called = match_function_call(line, fortran_names, functions)
                if len(functions_called) > 0:
                    if current_routine=="":
                        print( "\nnot in a routine, but it looks like some function(s) are called:\n", functions_called)
                        print( "call not added to the tree. file=",filename,"line=")
                        print( line )
                    else:

                        for func in functions_called:

                            if func not in multi_named_routines.keys() or multi_named_routines[func] == True:
                                func_fullname = func
                            else:
                                found_fullname = False
                                for fullname in routines.keys():
                                    if func in re.split('\W+',fullname):
                                        func_fullname = fullname
                                        found_fullname = True
                                if found_fullname == False :
                                    print('Fuuuuck function!  ' , func, func in multi_named_routines.keys(), func in routines.keys())

                continue # no need to go on



        ## line done, reset
        long_line=""

    if pr: print('includes_list for ',filename,' : ',includes_list)
# end check_for_includes

#======================================================

# scan line by line for modules, routines, program etc
def prelim_scan_file(filename,type):
    
    #srclines = out.split('\n')
    srclines = copy.copy(source_code_dict[filename][0])
    
    long_line = ""      # the line that will be sent to the splitter
    stored_line = ""    # the statement following long_line
    last_line=0         # if true, fline is the last line of a file.

    in_module_bool = False  # switch to true when we are in a module
    in_routine     = 0      # increment by 1 each time we enter, and vice versa
    position = [('','')]    # add subroutine/module name to list when we enter it
    position_bkp = []       # bkp list to deal with cpp statements
    parent_name  = ''       # will hold name of either subroutine or module in which a found routine is CONTAINed, if it is the case
    in_contains_bool = False
    in_explicit_interface = False

    in_interface = False
    interfaced_procedures = []
    interface_name = ""

    pr = False
    #if 'gprcp.F90' in filename: pr = True  # switch on debugging printing
    if pr: print('looking at ',filename,' in prelim_Scan')

    current_host=""

    for line_num,fline in enumerate(srclines):

        ## the scanner. feed each line into the joiners

        if type=="free":
            long_line, get_next = join_and_strip_free(fline, long_line)

        elif type in ["fixed","fixed132"]:
            last_line = cline==len(srclines)
            long_line, get_next, stored_line, warning_flag = join_and_strip_fixed(\
                fline, long_line, stored_line, last_line, type=type)
        else:
            long_line, get_next = fline.strip(), 0

        if get_next:
            continue  ## break out, back to loop for with next fline


        long_line = strip_strings(long_line)

        ## the splitter. split each line at ;
        ## this gives us one statement per line
        # incidentally, THIS SHOULD NOT BE COMMON IN THE IFS

        for line in long_line.split("$$$$$"): # there are some printed strings that have ; in them, so can't just split on this ...
            line = line.strip()
            if line == "":
                continue
            #lines += 1

            line = line.upper()  ## shift all characters to UPPERCASE

            # we are now left with very naked fortran

            # ignore lines starting with comments ...
            if line.startswith('!') : 
                continue

            fortran_names = match_fortran_name.findall(line)  ## list of Fortran words (function name, variables, module name, etc) in line

            # deal with preprocessor lines
            if line.startswith('#IF'):
                position_bkp.insert(0,copy.deepcopy(position)) # keep a copy of original position in its state at #if
                if pr: print('backing up position for IFDEF')
            elif line.startswith('#EL') or line.startswith('#END'):
                if pr: print('going to restore position : ',position, position_bkp)
                try:
                    position = copy.deepcopy(position_bkp[0])  # restore the original from the copy
                except : 
                    print('deep copy didnt work, look for #EL or #END in ',filename)
                    sys.exit()
                if line.lstrip().startswith('#END'):
                    del(position_bkp[0])

            if line.strip() == 'INTERFACE' :
                in_explicit_interface = True
            elif len(fortran_names) == 2 :
                if fortran_names[0] == 'END' and fortran_names[1] == 'INTERFACE':
                    in_explicit_interface = False
                    
                    
            ## first look for MODULE definitions
            name = match_module(fortran_names) ## return name of module if line contains a declaration, or empty string otherwise             
            if name != "":   ## we have a module declaration on this line
                in_module_bool = True
                position.append( (re.split('\W+',line)[1] ,'module') )  # was long_line
                parent_name = re.split('\W+',line)[1]  + '%'            # was long_line
                if pr: print('found module : ',position,line)
                continue


            if len(position) == 0:
                print('shhhit ...',filename, line)
                sys.exit()

            if position[-1][0] != '':
                parent_name = position[-1][0]+'%'
            else:
                parent_name = ''

            ## look also for INTERFACES with MODULE PROCEDUREs
            interface_name, interfaced_procedures,in_interface =  match_module_procedure( 
                                fortran_names,line,srclines,parent_name, interface_name, interfaced_procedures,in_interface ) 
            if interface_name != "" and interface_name not in ['OPERATOR', 'ASSIGNMENT']: # this is an interface for routines, so add it to dictionary
                in_routine_, current_routine = add2dict(routines, interface_name,
                              (filename, [], []))

            ## look for PROGRAM definition
            name = match_program_def(fortran_names)
            if name != "":      ## we've found one 

                in_program, current_program = add2dict(routines, name, (filename, [], []))
                in_program, current_program = add2dict(programs, name, (filename, [], []))
                register_main_program(filename)

                position.append( (name,'program') )
                if pr: print('found PROGRAM', position)
                parent_name = name  + '%'
                continue



            ## in this pass the main task is to find the routine
            ## names, and the mapping between routine name and the
            ## file name. we should also figure out which
            ## namespace a routine name belongs to. we have the
            ## global namespace, and namespaces for modules and
            ## routines.


            ## look for SUBROUTINE definition
            name = match_subroutine_def(fortran_names)
            # if in a module or subprog, add name to local namespace, otherwise to the global.
            if name!="":  ## we've found one
                # if we're not in_routine or in_module , parent_NAME = False
                if position != [('','')] and not in_explicit_interface:
                    parent_name = position[-1][0]+'%'
                    if pr: print('is this the problem? position=',position,', parent_n=', parent_name,', name=',name)
                else:
                    parent_name = ''
                in_routine_, current_routine = add2dict(routines, parent_name+name, (filename, [], []))
                if pr: print('found inital scan subroutine def: ',parent_name + name, position, line)

                if name in multi_named_routines.keys():
                    multi_named_routines[name] = True # nth occurrence of this routine name
                else :
                    multi_named_routines[name] = False # first occurrence of this routine name
                in_routine += 1

                if in_module_bool : # this is a subroutine contained in a module
                    in_contains_bool = True
                else: # not in a module
                    if position == [('','')]:  # not a contained routine
                        in_contains_bool = False

                    else:  # this should be a subroutine CONTAINed in a subroutine
                        in_contains_bool = True
                        #parent_name = parent_list[-1] + '%'


                position.append( (name,'routine') )
                if pr: print('found routine', position)
                #parent_name = ''

                continue


            ## look for function definition 
            name = match_function_def(fortran_names,line)
            if name!="":
                if position != [('','')]:
                    parent_name = position[-1][0]+'%'
                else:
                    parent_name = ''
                in_routine, current_routine = add2dict(routines, parent_name+name, (filename, [], []))
                in_routine, current_routine = add2dict(functions, parent_name+name, (filename, [], []))
                position.append( (name,'function') )
                if pr: print ('found and added function: ',parent_name+name,position)


                if name in multi_named_routines.keys():
                    multi_named_routines[name] = True # nth occurrence of this routine name
                else :
                    multi_named_routines[name] = False # first occurrence of this routine name
                in_routine += 1
                if in_module_bool : # this is a subroutine contained in a module
                    in_contains_bool = True
                else: # not in a module
                    if position == [('','')]:  # not a contained routine
                        #in_contains_bool = False
                        in_contains_bool = False

                    else:  # this should be a subroutine CONTAINed in a subroutine
                        in_contains_bool = True
                        #parent_name = parent_list[-1] + '%'
                continue


            ## look for end of module
            name = match_module_end(fortran_names)
            if name != "":
                current_host=""
                contained=0
                module_includes = []

                if in_module_bool == True: 
                    in_module_bool = False
                    position = [('','')]
                    if pr: print('found END MOD', position)
                    in_contains_bool = False
                else: # we seem to have found an END MODULE with no corresponding start ...
                    raise SystemError('check '+filename+' for END MODULE with no MODULE to start it ...')
                continue


            ## look for end of sub(routine/function/program)
            name = match_end_subprogram(fortran_names)
            #if pr: print(line_num,line)
            if name != "":
                in_routine -= 1

                if in_module_bool :  # this was a contained subroutine
                    if (position[-1][1] in ['routine','function'] ) :
                        del(position[-1])
                        if pr: print('found end routine/function', position)
                    else:
                        print('problem : ',position, line,name,filename)
                        raise SystemError('exiting ...')
                else: # this was not in a module
                    if in_contains_bool :
                        del(position[-1])  # delete the last element from the list
                        if pr: print('found end routine/function', position)
                    else: # this routine was not contained, either by a module or by another routine
                        if position != [('','')]:
                            position = [('','')]
                            parent_name = ''
                        else: # we seem to have found an END SUBROUTINE with no corresponding start ...
                            print('check '+filename+' for END SUBROUTINE/FUNCTION with no SUBROUTINE to start it ...',long_line,position)
                            raise SystemError('exiting ...')
                if pr: print ('end of routine, remaining position: ',position, line)
                continue


        ## line done, reset
        long_line=""
    #if pr: sys.exit() 

# end prelim_scan_file    

#======================================================

# scan line by line for modules, routines, program etc
def scan_multiple_standalones_per_file(filename,type, ignore_these_multiples):
    
    #srclines = copy.copy(source_code_dict[filename])
    srclines = source_code_dict[filename][0]
    
    long_line = ""      # the line that will be sent to the splitter
    stored_line = ""    # the statement following long_line
    last_line=0         # if true, fline is the last line of a file.
    num_routines_in_file = 0

    in_module_bool = False  # switch to true when we are in a module
    in_routine     = 0      # increment by 1 each time we enter, and vice versa
    position = [('','')]    # add subroutine/module name to list when we enter it
    position_bkp = []       # bkp list to deal with cpp statements
    parent_name  = ''       # will hold name of either subroutine or module in which a found routine is CONTAINed, if it is the case
    in_contains_bool = False
    in_explicit_interface = False

    in_interface = False
    interfaced_procedures = []
    interface_name = ""

    pr = False
    #if 'gprcp.F90' in filename: pr = True  # switch on debugging printing
    if pr: print('looking at ',filename,' in prelim_Scan')

    for line_num,fline in enumerate(srclines):

        ## the scanner. feed each line into the joiners

        if type=="free":
            long_line, get_next = join_and_strip_free(fline, long_line)

        elif type in ["fixed","fixed132"]:
            last_line = cline==len(srclines)
            long_line, get_next, stored_line, warning_flag = join_and_strip_fixed(\
                fline, long_line, stored_line, last_line, type=type)
        else:
            long_line, get_next = fline.strip(), 0


        if get_next:
            continue  ## break out, back to loop for with next fline


        long_line = strip_strings(long_line)

        ## the splitter. split each line at ;
        ## this gives us one statement per line
        # incidentally, THIS SHOULD NOT BE COMMON IN THE IFS

        for line in long_line.split("$$$$$$"):
            line = line.strip()
            if line == "":
                continue
            #lines += 1

            line = line.upper()  ## shift all characters to UPPERCASE

            # we are now left with very naked fortran

            # ignore lines starting with comments ...
            if line.startswith('!') : 
                continue

            fortran_names = match_fortran_name.findall(line)  ## list of Fortran words (function name, variables, module name, etc) in line

            # deal with preprocessor lines
            if line.startswith('#IF'):
                position_bkp.insert(0,copy.deepcopy(position)) # keep a copy of original position in its state at #if
                if pr: print('backing up position for IFDEF')
            elif line.startswith('#EL') or line.startswith('#END'):
                if pr: print('going to restore position : ',position, position_bkp)
                try:
                    position = copy.deepcopy(position_bkp[0])  # restore the original from the copy
                except : 
                    print('deep copy didnt work, look for #EL or #END in ',filename)
                    sys.exit()
                if line.lstrip().startswith('#END'):
                    del(position_bkp[0])

            if line.strip() == 'INTERFACE' :
                in_explicit_interface = True
            elif len(fortran_names) ==2 :
                if fortran_names[0] == 'END' and fortran_names[1] == 'INTERFACE':
                    in_explicit_interface = False
                    
                    
            ## first look for MODULE definitions
            name = match_module(fortran_names) ## return name of module if line contains a declaration, or empty string otherwise             
            if name != "":   ## we have a module declaration on this line
                in_module_bool = True
                position.append( (re.split('\W+',line)[1] ,'module') )  # was long_line
                parent_name = re.split('\W+',line)[1]  + '%'            # was long_line
                if pr: print('found module : ',position,line)
                continue


            if len(position) == 0:
                print('shhhit ...',filename, line)
                sys.exit()

            if position[-1][0] != '':
                parent_name = position[-1][0]+'%'
            else:
                parent_name = ''


            ## look for PROGRAM definition
            name = match_program_def(fortran_names)
            if name != "":      ## we've found one 

                position.append( (name,'program') )
                if pr: print('found PROGRAM', position)
                parent_name = name  + '%'
                continue




            ## look for SUBROUTINE definition
            name = match_subroutine_def(fortran_names)
            # if in a module or subprog, add name to local namespace, otherwise to the global.
            if name!="":  ## we've found one
                # if we're not in_routine or in_module , parent_NAME = False
                if position != [('','')] and not in_explicit_interface:
                    parent_name = position[-1][0]+'%'
                    if pr: print('is this the problem? position=',position,', parent_n=', parent_name,', name=',name)
                else:
                    parent_name = ''
                    num_routines_in_file += 1
                    if not any(nme in filename for nme in ignore_these_multiples):
                        standalone_routines[name] = []
                in_routine += 1

                if in_module_bool : # this is a subroutine contained in a module
                    in_contains_bool = True
                else: # not in a module
                    if position == [('','')]: # not a contained routine
                        in_contains_bool = False

                    else:  # this should be a subroutine CONTAINed in a subroutine
                        in_contains_bool = True


                position.append( (name,'routine') )
                if pr: print('found routine def : ', position)
                #parent_name = ''

                continue



            ## look for function definition 
            name = match_function_def(fortran_names,line)
            if name!="":
                if position != [('','')]:
                    parent_name = position[-1][0]+'%'
                else:
                    parent_name = ''
                position.append( (name,'function') )
                if pr: print ('found function: ',parent_name+name,position)


                in_routine += 1
                if in_module_bool : # this is a subroutine contained in a module
                    in_contains_bool = True
                else: # not in a module
                    if position == [('','')]:  # not a contained routine
                        in_contains_bool = False

                    else:  # this should be a subroutine CONTAINed in a subroutine
                        in_contains_bool = True
                        #parent_name = parent_list[-1] + '%'
                continue


            ## look for end of module
            name = match_module_end(fortran_names)
            if name != "":
                current_host=""
                contained=0

                if in_module_bool == True: 
                    in_module_bool = False
                    position = [('','')]
                    if pr: print('found END MOD', position)
                    in_contains_bool = False
                else: # we seem to have found an END MODULE with no corresponding start ...
                    raise SystemError('check '+filename+' for END MODULE with no MODULE to start it ...')
                continue


            ## look for end of sub(routine/function/program)
            name = match_end_subprogram(fortran_names)
            #if pr: print(line_num,line)
            if name != "":
                in_routine -= 1

                if in_module_bool :  # this was a contained subroutine
                    if (position[-1][1] in ['routine','function'] ) :
                        del(position[-1])
                        if pr: print('found end routine/function', position)
                    else:
                        print('problem matching end of routine to a start : ',position, line,name,filename)
                        raise SystemError('exiting ...')
                else: # this was not in a module
                    if in_contains_bool :
                        del(position[-1])  # delete the last element from the list
                        if pr: print('found end routine/function', position)
                    else: # this routine was not contained, either by a module or by another routine
                        if position != [('','')]:
                            position = [('','')]
                            parent_name = ''
                        else: # we seem to have found an END SUBROUTINE with no corresponding start ...
                            print('check '+filename+' for END SUBROUTINE/FUNCTION with no SUBROUTINE to start it ...',long_line,position)
                            raise SystemError('exiting ...')
                if pr: print ('end of routine, remaining position: ',position, line)
                continue


        ## line done, reset
        long_line=""
    #if pr: sys.exit() 
    if num_routines_in_file > 1 and not any(nme in filename for nme in ignore_these_multiples):
        print(' case of multiple standalones : ',filename)

# end scan_multiple_standalones_per_file 


#======================================================

def strip_comments(line):
    """
    line is one line of free format Fortran. return the line
    stripped for comments as well as leading and trailing white
    space.
    """
    line = line.lstrip()             # remove leading spaces
    ccol = line.find("!")            # find position of first !
    # TODO: ! could be inside a string...
    if ccol<0:
        return line
    if ccol==0:
        return ""                    # it was a comment line
    if ccol>0:
        return line[:ccol].rstrip()  # remove everything after !


#======================================================

def match_module(names):
    """
    line is a stripped Fortran statement. (no comment or
    white space at beginning or end.)  if it contains the beginning of
    a module definition, return the module name in upper case
    """
    if len(names) < 2 : return ""
    if names[0] == "MODULE" and names[1] != "PROCEDURE":
        return names[1]
    return ""


#======================================================


def match_module_end(names):
    """
    """
    if len(names) < 3: return ""
    if names[0]=="END" and names[1]=="MODULE":
        return names[2]
    return ""


#======================================================


def match_end_subprogram(names):
    """
    """
    if len(names) > 2 :
        if names[0]=="END" and names[1] in ["SUBROUTINE","FUNCTION","PROGRAM"]:
            return names[2]
    if len(names) == 2 :
        if names[0] in ["ENDSUBROUTINE","ENDFUNCTION","ENDPROGRAM"]:
            return names[1]
        elif names[0] == 'END' and names[1] in ['FUNCTION','SUBROUTINE','PROGRAM'] :
            return names[1]+" without closing name"
    if len(names) == 1:
        if names[0] == "END":
            return "OLDSTYLE_END"
    return ""


#======================================================


def match_end(names):
    """
    """    
    if len(names) == 1:
        if names[0]=="END":
            return 1
    return 0


#======================================================


def match_subroutine_end(names):
    """
    """
    if len(names) < 3: return ""   
    if names[0]=="END" and names[1]=="SUBROUTINE":
        return names[2]
    return ""


#======================================================


def match_function_end(line):
    """
    """
    if len(names) < 3: return ""
    if names[0]=="END" and names[1]=="FUNCTION":
        return names[2]
    return ""


#======================================================


def match_program_def(names):
    """
    """
    if len(names) < 2 : return ""
    if names[0]=="PROGRAM":
        return names[1]
    return ""


#======================================================


def match_and_chop(line, pos, delimiter,orig_line):
    """
    line has delimiter at pos, find its match, and return line
    without the contained string (also remove the delimiters)
    """

    debug = False
    
    #mpos = line[pos+1:].find(delimiter) # we want to find next *single* delimiter
    templine =copy.copy(line)
    while delimiter+delimiter in templine[pos+1:]:
        templine = templine[:pos+1] + templine[pos+1:].replace(delimiter+delimiter,"",1)
    mpos =  templine[pos+1:].find(delimiter) 
    
    if mpos>0:
        if debug: print('should bleddy well print this')
        if pos+1+mpos < len(templine)-1:                                       # was line rather than templine
            if templine[pos+mpos+2]==delimiter:                                # was also line
#                return line[:pos+mpos+1]+line[pos+mpos+3:]
                return templine[:pos+mpos+1]+"A_STRING"+templine[pos+mpos+3:]  # was also line
            else:
#                return line[:pos]+line[pos+1+mpos+1:]
                return templine[:pos]+"A_STRING"+templine[pos+1+mpos+1:]       # was also line
        else:
#            return line[:pos]+line[pos+1+mpos+1:]
            return templine[:pos]+"A_STRING"+templine[pos+1+mpos+1:]           # was also line
    elif mpos==0:
        return templine[:pos] + "A_STRING" + templine[pos+2:]                  # was also line
    else:
        print( "error empty_string: only found one " + delimiter + " in line:")
        raise SystemError( templine+' ===== '+orig_line )
        sys.exit(2)


#======================================================


def strip_strings(line):
    """
    line is a stripped (python wise) fortran statement.  return line, with 
    all fortran strings removed, delimiters included...
    """
    orig_line = copy.deepcopy(line) # backup of orig line for debugging purposes
    
    debug = False
        
    if "! -1 = Couldn't open file" in line:
        dqpos=line.find('"')
        sqpos=line.find("'")
        expos=line.find("!")
        
    quotes_in_string=True
    while quotes_in_string:
        dqpos=line.find('"') # dq : double quote
        sqpos=line.find("'") # sq : single quote
        expos=line.find("!") # ex : exclamation
        if dqpos<0 and sqpos<0:
            quotes_in_string=False
        elif expos > -1 and (sqpos < 0 or sqpos > expos) and (dqpos < 0 or dqpos > expos) : # exclamation before quotes : definitely a comment
            return line.split('!')[0]
        else:
            if dqpos<0: # we must therefore have single quotes
                if len(re.findall("'",line))==1 : print('this is the culprit! ' , line)
                line = match_and_chop(line, sqpos, "'",orig_line)
                quotes_in_string = ("'" in line)
                if debug: print('strip_strings received this back from match_and_chop : \n',line,'\nand quotes_in_string = ',quotes_in_string)
            else:
                if sqpos<0:
                    line = match_and_chop(line, dqpos, '"', orig_line)
                else:
                    if dqpos<sqpos:
                        line = match_and_chop(line, dqpos, '"', orig_line)
                    else:
                        line = match_and_chop(line, sqpos, "'", orig_line)
    return line

#end strip_strings

#======================================================


def match_function_def(names,line):
    """
    """
    
    try:
         sl = strip_strings(line)
    except:
        print('bugging line :',line)
    if 'FUNCTION' in sl:
        # double precision function xxx
        if len(names) >=4:
            for nn in range(2,4):
                if names[nn]=="FUNCTION":
                    return names[nn+1]    

        if len(names) >=3:
            if names[1]=="FUNCTION" and names[0]!="END":
                return names[2]

        if len(names) >=2 :
            if names[0]=="FUNCTION":
                return names[1]

    return ""

#end match_function_def

#======================================================


def match_subroutine_def(names):
    """
    """
    if len(names) < 2 : return ""
    if names[0]=="SUBROUTINE":
        return names[1]
    elif names[0] in ["RECURSIVE","ELEMENTAL","PURE"] and names[1] == "SUBROUTINE":
        return names[2]
    return ""

#end match_subroutine_def

#======================================================

def match_subroutine_call(names, lne='',prevlines = ['']):
    """
    """
    if len(names) < 1: return ""
    try:
        i=names.index("CALL")
    except ValueError:
        return ""

    uplne = lne.upper()

    after_call = uplne.split('CALL ')[1].lstrip()
    after_call_words = re.split( '\W+', after_call )
#    if lne != '' and lne.split('CALL ')[1].lstrip().startswith('%') : # 'CALL' is in the line, we would have returned already, otherwise
    if uplne != '' and len(after_call_words)>1:
        if after_call.split(  after_call_words[0] )[1][0]=='%'  : # 'CALL  XXX%ROUT' : ROUT is a type-bound routine
            # probably should try to find the type to which we are bound ...'
            type_name = ''
            lnenum = len(prevlines)-1
            while lnenum >=0  :
                words = re.split( '\W+', prevlines[lnenum].strip().split('!')[0].upper() )
                if (all( wd in words for wd in ['TYPE' , names[i+1]] ) or \
                    all( wd in words for wd in ['CLASS' , names[i+1]] ) ): # TYPE and instance name are in this line
                    
                    if words[0] == 'TYPE' : 
                        type_name = words[ words.index('TYPE')+1 ]
                    else :
                        try :
                            type_name = words[ words.index('CLASS')+1 ]
                        except :
                            print('???????? ',uplne,'\n',words,'\n',prevlines[lnenum],prevlines)
                            raise SystemError('#####')
                    break
                elif all( wd in words for wd in ['USE' , names[i+1]] ) : # a singleton derived type object with type-bound method being called
                    type_name = after_call_words[0]
                    print('found example of singleton derived type with bound method being called : \n',uplne)
                    break
                lnenum -= 1
            if type_name == '' :
                lnenum = 0
                while lnenum < len(prevlines) :
                    if not prevlines[lnenum].lstrip().startswith('!') and 'ATLAS_MODULE' in prevlines[lnenum].upper() : # horrible hack : for the moment, assume that type will come from ATLAS
                        type_name = 'ATLAS_MODULE'
                        #print('default attribution of atlas_module type to type-bound method ',names[i+2])
                        #input('#####')
                        break
                    lnenum += 1
                        
            if type_name == '' :
                print('damn, didnt find the type for this line : \n',lne,'\n',prevlines)
                raise SystemError('###')
            else:
                # now look for module name
                mod_name = ''
                lnenum2 = 0
                while lnenum2 <= lnenum :  # the module has to have been USEd before the declaration ...
                    if (not prevlines[lnenum2].lstrip().startswith('!') and all( wd in prevlines[lnenum2].upper() for wd in ['USE' , type_name] )) \
                       or 'ATLAS_MODULE' in prevlines[lnenum2].upper() : # USE and type name are in this line
                        words2 = re.split( '\W+', prevlines[lnenum2].upper() )
                        try : 
                            mod_name = words2[ words2.index('USE')+1 ]
                        except :
                            print('why is this bloody failing??? ',prevlines[lnenum2])
                            input('###########')
                        break
                    elif  not prevlines[lnenum2].lstrip().startswith('!') and all( wd in prevlines[lnenum2].upper() for wd in ['PROCEDURE', names[i+2]]):
                        # mod_name is the name of the module we're looking at right now
                        lnenum3 = copy.copy(lnenum2)
                        while not prevlines[lnenum3].lstrip().upper().startswith('MODULE') and lnenum3 > 0: # find the line declaring the module
                            lnenum3 -= 1
                        if not prevlines[lnenum3].lstrip().upper().startswith('MODULE') : 
                            print('dammmmmmit it failed! ',names[i+2],'\n',prevlines)
                            input('*********')
                        else :
                            mod_name = re.split('\W+', prevlines[lnenum3])[1]
                        break

                    lnenum2 += 1
                if mod_name == '':
                    print('feeehhhck ! havent found the module?? ',lne,type_name,lnenum,'\n',prevlines[:lnenum])
                    raise SystemError(';;;')

            #print('looks like we have found a type-bound routine!',lne, '---',names[i+2],'---',type_name,'---',mod_name)
            #input('%%%')
            return mod_name+'%'+names[i+2]
        
    return names[i+1]  # the default return if we find a subroutine

#end match_subroutine_call

#======================================================

def match_subroutine_call_simple(names, lne='',prevlines = [''], fname=''):
    """
    """
    if len(names) < 1: return ""
    try:
        i=names.index("CALL")
    except ValueError:
        return ""

    try : 
        uplne = strip_comments( strip_strings( lne.upper() ) )
    except :
        print('line causing sys error : ', lne)
        sys.exit()
    local_words = re.split('\W+',uplne)
    if 'CALL' not in local_words or lne.startswith('#'):
        return ""

    after_call = uplne.split('CALL ')[1].lstrip()
    after_call_words = re.split( '\W+', after_call )
    # ignore type-bound routines for the moment because don't know how to do the regex, but look up later ...
    #if re.find() 
    
    if uplne != '' and len(after_call_words)>=1:
        #if after_call.split(  after_call_words[0] )[1][0]=='%'  : # 'CALL  XXX%ROUT' : ROUT is a type-bound routine        
        if '%' in after_call.split(  after_call_words[0] )[1]  : # 'CALL  XXX%ROUT' : ROUT is a type-bound routine
            if len( after_call.split(  after_call_words[0] )[1] ) == 0 :
                input('is this working? '+ uplne )
            else :
                return ""
                
    return names[i+1]  # the default return if we find a subroutine

#end match_subroutine_call_simple

#======================================================


def match_function_call(line, names, functions):
    """
    """
    functions_called=[]
    for name in names:
        if name in functions: ## functions.has_key(name):
            # ok, possible call. but we must check wether it is a call
            # or a local variable. we assume functions are called with
            # an argument list. this is not always the case.  if this
            # is not enough, we must know if there is a local variable
            # with the function name, which requires parsing...
            pattern = name + " *\("
            if len(re.findall(pattern, line)) >= 1 and '::' not in line :
                functions_called.append(name)
    return functions_called

#end match_function_call


#======================================================


def match_include(line, words, include_path=["."]):
    """
    match fortran and c include statements on a stripped line,
    return 1,filename for a match, 0,"" otherwise.
    """

    if words[0] != 'INCLUDE' :
        return ""
    if not line.lstrip().startswith('#') :
        return ""

    ## locate the quotes, that can be ' or ". filename inbetween.
    #firstquote = line[icol+8:].find('"')
    #if firstquote>-1:
    #    lastquote = line[icol+8:].rfind('"')
    #else:
    #    firstquote = line[icol+8:].find("'")
    #    if firstquote>-1:
    #        lastquote = line[icol+8:].rfind("'")
    #    else:
    #        print('an #include without quote marks???? ',line)
    #        sys.exit('this calls for an abort ...')
    #        return ""
    #
    #name=line[icol+9+firstquote:icol+8+lastquote]

    firstquote = line.find('"')
    if firstquote>-1:
        lastquote = line.rfind('"')
    else:
        firstquote = line.find("'")
        if firstquote>-1:
            lastquote = line.rfind("'")
        else:
            print('an #include without quote marks???? ',line)
            sys.exit('this calls for an abort ...')
            return ""

    name=line[firstquote+1:lastquote]
    if '.' not in name:
        print('we have a dodgy interface include with no .h ending? ',line)
        input('...')
    else :
        name = name.split('.')[0]
    return name

#end match_include


#======================================================


def match_module_procedure(words,line,flines,parent_name,  interface_name_input,interfaced_procedures_input,in_interface_input):
    """
    """
    interface_name = ""
    interfaced_procedures = []
    in_interface = False
    
    if len(words) >= 3 and words[0] == 'MODULE' and words[1] == 'PROCEDURE' and in_interface_input == False: # we've found one, now deal with it

        pos = 0
        pos_in_file = 0
        found = False

        iter_lines = iter(flines)
        for tline in iter_lines:

            # deal with continued lines
            long_line,orig_lines = assemble_continued_statement_from_iterator(tline,iter_lines)

            #if long_line.upper().strip() == line.strip():
            if match_fortran_name.findall(long_line.upper().split('!')[0]) == words:
                found = True
                pos_in_file = pos
            pos += len(orig_lines) # increment only if we haven't found the matching line


        interface_pos = pos_in_file-1
        while flines[interface_pos].upper().strip() == '' or flines[interface_pos].upper().strip().startswith('!'):
            print('shouldnt happen ',parent_name)
            interface_pos -= 1
        interface_line = flines[interface_pos].upper().strip()
        if not interface_line.startswith('INTERFACE'):
            print ('match_module_procedure : this should be interface for :',words,line,'...',interface_line,'...' ,interface_pos,pos_in_file-1,len(flines),flines[0])
            sys.exit()
        else:
            interface_name = match_fortran_name.findall(interface_line)[1]
            if interface_name in ['OPERATOR', 'ASSIGNMENT'] :  # short-circuit exit : we are ignoring these cases
                return interface_name,interfaced_procedures,True #in_interface = True
            
            if parent_name != '':
                interface_name = parent_name + interface_name
                #print('found a module procedure interface with a parent_name : ',interface_name)
            
            in_interface = True
            interfaced_procedures = [parent_name+wd for wd in words[2:]]

    elif in_interface_input == True and words[0] == 'MODULE' and words[1] == 'PROCEDURE'  and len(words) >= 3 : # we've found another one
        interfaced_procedures = interfaced_procedures_input + [parent_name+wd for wd in words[2:]]
        interface_name = interface_name_input
        in_interface = True
        
    elif in_interface_input == True and words[0] == 'END' and words[1] == 'INTERFACE'  : # end of the interface
        in_interface = False
        
    elif 'MODULE' in words and 'PROCEDURE' in words:
        print ('what is going on? ',line,words)
        sys.exit()


    return interface_name,interfaced_procedures,in_interface

#end match_module_procedure

#======================================================


def join_and_strip_free(line, joined_line):
    """
    line is a free format fortran line from the scanner.
    if line has an ampersand at the beginning, append line to joined line.
    if line has an ampersand at the end, return the joined line and a flag=1.
    otherwise flag=0.
    """
    ## strip off free form comments
    ## this is safe(r) if we have an even number of " or ' up until the !

    ## fails on
    ##   write(sf,'("!! -*-f90-*-",/,"!!",/,a)', advance='no') "!! Restart file written by BOM5 on "

    expos = line.find("!")


    if expos >= 0:
        # figure out if it is safe to run the brute force
        # strip_comments by counting quotes. if the string before the
        # ! has uneven number of single OR double quotes it is not
        # safe, because ! probably is in a string.
        ns1=count_single_quotes(joined_line)
        ns2=count_single_quotes(line[:expos])
        nd1=count_double_quotes(joined_line)
        nd2=count_double_quotes(line[:expos])

        unsafe = (ns1+ns2)%2 or (nd1+nd2)%2
        
        if not unsafe:            
            line = strip_comments(line)

        
    if line=="":
        return joined_line,1

    ## remove ampersand at beginning   
    line=line.lstrip()
    if line!="":
        if line[0]=="&":
            line=line[1:]

    ## then check for ampersand at the end
    amppos = line.rfind("&")

    if amppos>-1:
        ## if there is something on the line after the last amp. it could mean the amp is in a string.
        if line[amppos+1:].strip() == "":
            joined_line += line[:amppos].strip() + " "
            return joined_line, 1
        else:
            joined_line += line.strip()    
    else:
        joined_line += line.strip()
    return joined_line, 0

#end join_and_strip_free

#======================================================


def handle_last(line, long_line, next_line):
    warning_flag=0
    if line.strip()=="":
        if long_line != "":
            return long_line, 0, "",warning_flag
        if next_line != "":
            return next_line, 0, "",warning_flag
        warning_flag=1
        return "",0,"",warning_flag
    if line[0] in ["C","c","!","*"]:
       if long_line!="":
            return long_line, 0, "",warning_flag
       if next_line!="":
            return next_line, 0, "",warning_flag
    contline=0
    if len(line)>6:
        if line[5]!=" ":
            contline=1
    line=line[6:]
    if contline:
        if long_line != "":
            long_line += line.strip()
        if next_line != "":
            long_line = next_line + line.strip()
        return long_line, 0, "",warning_flag
    if not contline:
        if long_line!="":
            return long_line + ";" + line.rstrip(), 0, "",warning_flag
        if next_line!="":
            return next_line+";"+line.rstrip(), 0, "",warning_flag
    return line.rstrip(), 0, "", warning_flag

#end handle_last


#======================================================


def count_quotes(string):
    """
    return the number of quotes in string
    """
    return len(match_single_quote.findall(string)) + len(match_double_quote.findall(string))


#======================================================


def count_single_quotes(string):
    """
    return the number of quotes in string
    """
    return len(match_single_quote.findall(string))


#======================================================


def count_double_quotes(string):
    """
    return the number of quotes in string
    """
    return len(match_double_quote.findall(string))
    


#======================================================


def strip_trailing_comment(line, n_prev=None):
    """
    line is the beginning of a valid fortran statement.
    we want to return the statement without the comment.
    we count ' and ", if we have an even number of both,
    we can be pretty sure a ! is a comment.

    if n_prev is present, it is the number of quotes from a previous
    string to be prepended
    """
    i=0     # position of current ! in the substring after the previous
    gi=0    # global position of previous !

    # when i is -1 we have tested all !'s
    while i >= 0:
        gi += i+1
        i = line[gi+1:].find("!")
        if i >=0:
            n = count_quotes(line[:gi+1+i])
            
            # if the string is a continuation, n_prev is the number of
            # quotes in the statement we are appending to
            if n_prev: n += n_prev
            
            if n>0:
                # if n is even (n%2==0) we can chop:
                if not n%2 :
                    return line[:gi+1+i]
            else:
                return line[:gi+1+i]
    return line

#end strip_trailing_comment

#========================================================================



def join_and_strip_fixed(line, long_line, next_line, last, type="fixed"):
    """
    fixed format is a bit more tedious than free form; when the scanner feeds us a
    line we dont know yet if the next line is a continuation line or not,

    we can therefore conclude that we have a complete line only when a new line is found.
    therefore the parsing of fixed form code is done with a one line lag
    w.r.t. the free form (that we started out with), so unfortunately
    this routine became a bit messy...

    line is the thing we get from the scanner.

    if this line is not a continuation line, long_line contains a line
    of fortran we can parse. we then signal that we want to parse
    long_line, and store "line" in "next_line"
    
    if line is a cont line, we must add it to long_line and say we
    need one more line. next_line is therefore set to "".

    return 1 as the second result if the scanner should fetch more lines, 0 if
    we have a complete line ready to be parsed.
    """


    # if 1st bit: tab on line, 2nd bit: possibly continuation over cpp
    warning_flag=0

    if line.find("\t")>-1:

        print( "Warning: this line has a TAB:")
        print( line )
        warning_flag |= 2**0


    ## special treatment of the last line in the file
    if last:
        a1,b1,c1,d1 = handle_last(line, long_line, next_line)
        return a1,b1,c1,d1

    ## if line is blank we should check if next_line or long_line has
    ## something to be written: if long_line is !="" we can send it to
    ## splitting, and set next_line="". next_line and long_line cannot
    ## normally be "" simultaneously since long_line is "" after the
    ## splitter. if next_line is!="" while line is blank it means we
    ## can send it to the splitter
    pr = False#True
 

    if line.strip()=="":
        if long_line != "":
            return long_line, 0, "",warning_flag
        if next_line != "":
            return next_line, 0, "",warning_flag
        return "", 1, "",warning_flag


    ## if we are a comment line, we may need to join more

    if line[0] in ["C","c","!","*"]:
        if pr: print('we have a comment ! ',long_line)
        return long_line, 1, next_line,warning_flag


    ## (cpp handling goes here if needed)
    if line[0] == '#' :
        #if line.startswith('#else') : print ('check #else: ', line,'...',long_line,'...',next_line)
        #if pr: print ('check #: ', line,'...',long_line,'...',next_line)
        if long_line!="":
            next_line=line.rstrip()
            return long_line, 1, next_line,warning_flag
        if next_line!="":
            long_line=next_line
            next_line=line.rstrip()
            return long_line, 1, next_line,warning_flag
        if long_line == '' and next_line == '': #temp test, *seems* to work ...
            return line,1,'',warning_flag
    
    ## strip stuff after col 72 for pure fixed form
    if type=="fixed":
        if len(line)>72 :
            line=line[:72].rstrip()  # changed from 73 to 72

    ## decide if "line" is a new line or a continuation line
    contline=0
    if len(line)>6:
        if line[5]!=" ":
            contline=1
            if pr: print('we have a continued line! ',line)
            
    ## at the very beginning of files, this may happen:
    if long_line==next_line=="":
        if line.strip()=="":
            return "",1,"",warning_flag
        return line[6:].rstrip(), 1, "",warning_flag

    line=line[6:]

    ## if we are a continuation line: if long_line!="" add to it.
    ## if next_line!="" make it long_line and add to that instead.
    ## return with next_line=""

    if contline:
        # try to remove trailing comments with the "' rule
        n = count_quotes(long_line)
        line = strip_trailing_comment(line, n)
        
        if long_line != "":
            long_line += " " + line.strip()
            if pr: print('continued line with ll!=""',long_line)
        if next_line != "":
            long_line = next_line + " " + line.strip()
            if pr: print('continued line with nl!=""',long_line)
        return long_line, 1, "",warning_flag

    ## if we are a new line: if long line!="", send it to splitter
    ## and return next_line=line
    ## if next_line!="", set long_line=next_line, next_line=line
    ## and return to splitter

    line = strip_trailing_comment(line)
    
    if long_line!="":
        next_line=line.rstrip()
        if pr: print('non-continued line with ll!=""',long_line)
        return long_line, 0, next_line,warning_flag
    if next_line!="":
        long_line=next_line
        next_line=line.rstrip()
        if pr: print('non-continued line with nl!=""',long_line)
        return long_line, 0, next_line,warning_flag
    
    print( "join_and_strip_fixed error. you should not see this message.")
    sys.exit(1)

#end join_and_strip_fixed


#======================================================


def assemble_continued_statement_from_iterator(line, flines,return_orig = True):
    # Remove any trailing comment on line, careful of exclamations in strings!
    # What to do about comments?
    #statement = line.split('!')[0]
    statement = line 

    # store the list of original lines composing the continued statement
    original_lines = [line]

    if line.startswith('#'):
        if return_orig : 
            return statement,original_lines
        else:
            return statement
    

    # for an empty line, nothing needs to be done 
    line_stripped = line.strip()
    if line_stripped != '':

        if not line_stripped.startswith('!$OMP'):

            if '&' in line_stripped and line_stripped[0] != '!':

                #m = re.match('(.*?)&', line.split('!')[0])  ## return the start of the string up to '&' if it is present
                if len(match_single_quote.findall(line.split('!')[0]))%2==0 and len(match_double_quote.findall(line.split('!')[0]))%2==0 : 
                    m = up_to_ampersand.match(line.split('!')[0])  ## return the start of the string up to '&' if it is present: re.match('(.*?)&', ...
                else : 
                    m = up_to_ampersand.match(line)  ## the ! is in a string ...
                nline = line

                ## we look for a "&", no single quote mark before it, and no trailing chars, so should be line continuation
                #if len(nline.split('&'))==0 or len(nline.split('&')[1])==0 : print ('AHA! ', nline)
 
                while m is not None and (nline[nline.rfind('&')+1:].strip() == '' or nline[nline.rfind('&')+1:].strip().startswith('!')) : 
                                    # and len(re.findall(r"'",nline.split('&')[0]))%2 != 1   

                    # Add line contents up to any '&' to statement
                    #statement = m.group(1)  ## first group from matching string
                    statement = statement[:statement.rfind('&')] 
                    # Read next line while dropping leading &
                    ### nline = re.match('( +&)?(.*)',fid.readline()).group(2)+'\n'
                    nline = flines.__next__()
                    while nline.strip().startswith('!') or nline.strip()=='': # REMOVE THE ''
                        if return_orig: original_lines.append(nline)
                        nline = flines.__next__()
                    if return_orig: original_lines.append(nline)


                    if len(nline.strip())==0:
                        raise SystemError('Shouldnt be here in assemble_, aborting ',statement,'...',nline)
                    
                    while (nline.strip())[0] == '!':
                        nline = flines.__next__()
                    nline = whole_cont_line.match(nline).group(2)+'\n'  ## replaced + by * in regex, to allow for '&' as first character : re.match( '( *&)?(.*)',...

                    statement = statement + nline.lstrip()
                    m = up_to_ampersand.match(statement)  ## re.match('(.*?)&'
        

        else:
            ## deal with OMP directives
            if '&' in line: # we have a multiline OMP directive
                statement = line.split('&')[0] 

                nline = flines.__next__()
                if return_orig: original_lines.append(nline)
                if nline.strip() == '' :
                    raise SystemError('Aborting here, nline.strip should not be empty ',file_name,'\n', line,nline,fid.readline() )
                while nline.lstrip().startswith('!') and not nline.lstrip().upper().startswith('!$OMP'):
                    if return_orig: original_lines.append(nline)
                    nline = flines.__next__()

                statement = statement + ' ' + re.split('\!\$OMP *&*', nline.lstrip())[1]
                #print ('check continued OMP statement: \n',statement,'\n',nline,'\n',re.split('\!\$OMP *&', nline.lstrip())[1] )
                while statement.count('&') > 0:
                    nline = flines.__next__()
                    if return_orig : original_lines.append(nline)
                    #print (nline,'\n')
                    statement = statement.split('&')[0] + re.split('\!\$OMP *&*', nline.lstrip())[1]
                #print ('final OMP statement: \n',statement)  


    # keep comment lines !
    if line.strip() != '':
        if line.strip()[0] == '!' and not line.lstrip().startswith('!$OMP'):
            statement = line


    if return_orig : 
        return statement,original_lines
    else:
        return statement

#end assemble_continued_statement_from_iterator

#======================================================


# @profile
def assemble_continued_statement_from_list(line_num, lines_list,return_orig = True):
    # Remove any trailing comment on line, careful of exclamations in strings!
    # What to do about comments?
    #statement = line.split('!')[0]

    next_line_num = line_num+1
    line = lines_list[line_num]
    statement = line

    if return_orig:
        # store the list of original lines composing the continued statement
        original_lines = [line]

    if line.startswith('#'):
        #print(' a problem?? ',line)
        #input('...')
        if return_orig : 
            return next_line_num,statement,original_lines
        else:
            return next_line_num,statement
    

    # for an empty line, nothing needs to be done 
    if line.strip() != '':

        if not line.lstrip().startswith('!$OMP'):

            if '&' in line and line.lstrip()[0] != '!':

                #m = re.match('(.*?)&', line.split('!')[0])  ## return the start of the string up to '&' if it is present
                if len(match_single_quote.findall(line.split('!')[0]))%2==0 and len(match_double_quote.findall(line.split('!')[0]))%2==0 : 
                    #m = up_to_ampersand.match(line.split('!')[0])  ## return the start of the string up to '&' if it is present: re.match('(.*?)&', ...
                    m = '&' in line.split('!')[0]
                else : 
                    #m = up_to_ampersand.match(line)  ## the ! is in a string ...
                    m = '&' in line
                nline = line

                ## we look for a "&", no single quote mark before it, and no trailing chars, so should be line continuation
                #if len(nline.split('&'))==0 or len(nline.split('&')[1])==0 : print ('AHA! ', nline)
 
                #while m is not None and (nline[nline.rfind('&')+1:].strip() == '' or nline[nline.rfind('&')+1:].strip().startswith('!')) : 
                #                    # and len(re.findall(r"'",nline.split('&')[0]))%2 != 1   
                while m is not False and (nline[nline.rfind('&')+1:].strip() == '' or nline[nline.rfind('&')+1:].strip().startswith('!')) : 
                                    # and len(re.findall(r"'",nline.split('&')[0]))%2 != 1   

                    # Add line contents up to any '&' to statement
                    #statement = m.group(1)  ## first group from matching string
                    statement = statement[:statement.rfind('&')] 
                    # Read next line while dropping leading &
                    ### nline = re.match('( +&)?(.*)',fid.readline()).group(2)+'\n'
                    nline = lines_list[next_line_num]
                    next_line_num += 1
                    while nline.lstrip().startswith('!') or nline.strip()=='': # REMOVE THE ''
                        if return_orig: original_lines.append(nline)
                        nline = lines_list[next_line_num]
                        next_line_num += 1 
                    if return_orig: original_lines.append(nline)


                    if len(nline.strip())==0:
                        raise SystemError('Shouldnt be here in assemble_, aborting ',statement,'...',nline)
                    
                    while (nline.lstrip())[0] == '!':
                        nline = lines_list[next_line_num]
                        next_line_num += 1 

                    nline = whole_cont_line.match(nline).group(2)+'\n'  ## replaced + by * in regex, to allow for '&' as first character : re.match( '( *&)?(.*)',...

                    statement = statement + nline.lstrip()
                    #m = up_to_ampersand.match(statement)  ## re.match('(.*?)&'
                    m = '&' in statement 
        

        else:
            ## deal with OMP directives
            if '&' in line: # we have a multiline OMP directive
                statement = line.split('&')[0] 

                nline = lines_list[next_line_num]
                next_line_num += 1 
                if return_orig: original_lines.append(nline)
                if nline.strip() == '' :
                    raise SystemError('Aborting here, nline.strip should not be empty ',file_name,'\n', line,nline,fid.readline() )
                while nline.lstrip().startswith('!') and not nline.lstrip().upper().startswith('!$OMP'):
                    if return_orig: original_lines.append(nline)
                    nline = lines_list[next_line_num]
                    next_line_num += 1 

                statement = statement + ' ' + re.split('\!\$OMP *&*', nline.lstrip())[1]
                #print ('check continued OMP statement: \n',statement,'\n',nline,'\n',re.split('\!\$OMP *&', nline.lstrip())[1] )
                while statement.count('&') > 0:
                    nline = lines_list[next_line_num]
                    next_line_num += 1 
                    if return_orig : original_lines.append(nline)
                    #print (nline,'\n')
                    statement = statement.split('&')[0] + re.split('\!\$OMP *&*', nline.lstrip())[1]
                #print ('final OMP statement: \n',statement)  


    # keep comment lines !
    if line.strip() != '':
        if line.strip()[0] == '!' and not line.lstrip().startswith('!$OMP'):
            statement = line


    if return_orig : 
        return next_line_num, statement,original_lines
    else:
        return next_line_num, statement

#end assemble_continued_statement_from_list

#======================================================
#======================================================


#def assemble_fixed_statement_from_list(line, long_line, next_line, last):
def assemble_fixed_statement_from_list(line_num, srclines, return_orig = True):

    line = srclines[line_num]
    long_line = ''
    stored_line = ''
    last = (line_num==len(srclines)-1) # True if we are dealing with last line
    next_line_num = line_num


    for num,fline in enumerate(srclines[line_num:]):

        last = (line_num+num==len(srclines)-1) # True if we are dealing with last line

        long_line, get_next, stored_line, warning_flag = join_and_strip_fixed(\
                                                                              fline, long_line, stored_line, last, type='fixed')

        if get_next :
            next_line_num += 1
            continue
        else: # we have a complete line
            return next_line_num , long_line

#end assemble_fixed_statement_from_list


#======================================================


def assemble_continued_statement(line, fid,file_name=''):
    # Remove any trailing comment on line, careful of exclamations in strings!
    # What to do about comments?



    #statement = line.split('!')[0]
    statement = line 

    # store the list of original lines composing the continued statement
    original_lines = [line]

    if line.startswith('#'):
        return statement,original_lines
    
    # for an empty line, nothing needs to be done 
    if line.strip() != '':

        if not line.lstrip().startswith('!$OMP'):

            if '&' in line and line.lstrip()[0] != '!':

                #m = re.match('(.*?)&', line.split('!')[0])  ## return the start of the string up to '&' if it is present
                if len(re.findall('"',line.split('!')[0]))%2==0 and len(re.findall("'",line.split('!')[0]))%2==0 : 
                    #m = up_to_ampersand.match(line.split('!')[0])  ## return the start of the string up to '&' if it is present  ## re.match('(.*?)&',...
                    m = ('&' in line.split('!')[0])
                else : 
                    #m = up_to_ampersand.match(line)  ## the ! is in a string ... re.match('(.*?)&',
                    m = ('&' in line)
                nline = line

                ## we look for a "&", no single quote mark before it, and no trailing chars, so should be line continuation
                #if len(nline.split('&'))==0 or len(nline.split('&')[1])==0 : print ('AHA! ', nline)
 
                while m is True and (nline[nline.rfind('&')+1:].strip() == '' or nline[nline.rfind('&')+1:].strip().startswith('!') ): 
                                    # and len(re.findall(r"'",nline.split('&')[0]))%2 != 1   

                    # Add line contents up to any '&' to statement
                    #statement = m.group(1)  ## first group from matching string
                    statement = statement[:statement.rfind('&')] 
                    # Read next line while dropping leading &
                    ### nline = re.match('( +&)?(.*)',fid.readline()).group(2)+'\n'
                    nline = fid.readline()
                    while nline.lstrip().startswith('!')  or nline.strip()=='':
                        original_lines.append(nline)
                        nline = fid.readline()

                    original_lines.append(nline)


                    if len(nline.strip())==0:
                        raise SystemError('Shouldnt be here, aborting ',statement,'...',nline)
                    
                    #while (nline.lstrip())[0] == '!':
                    #    nline = fid.readline()
                    nline = whole_cont_line.match(nline).group(2)+'\n'  ## replaced + by * in regex, to allow for '&' as first character : re.match('( *&)?(.*)',...

                    statement = statement + nline.lstrip()
                    #m = up_to_ampersand.match(statement)  ## re.match('(.*?)&',
                    if '!' in statement:
                        m = ('&' in remove_trailing_comment(statement))
                    else:
                        m = ('&' in statement)


        else:
            ## deal with OMP directives
            if '&' in line: # we have a multiline OMP directive
                statement = line.split('&')[0] 

                nline = fid.readline()
                original_lines.append(nline)
                if nline.strip() == '' :
                    raise SystemError('Aborting here, nline.strip should not be empty ',file_name,'\n', line,nline,fid.readline() )
                while nline.lstrip().startswith('!') and not nline.lstrip().upper().startswith('!$OMP'):
                    nline = fid.readline()
                    original_lines.append(nline)
                    
                statement = statement + ' ' + re.split('\!\$OMP *&*', nline.lstrip())[1]
                #print ('check continued OMP statement: \n',statement,'\n',nline,'\n',re.split('\!\$OMP *&', nline.lstrip())[1] )
                while statement.count('&') > 0:
                    nline = fid.readline()
                    original_lines.append(nline)
                    #print (nline,'\n')
                    statement = statement.split('&')[0] + re.split('\!\$OMP *&*', nline.lstrip())[1]
                #print ('final OMP statement: \n',statement)  


    # keep comment lines !
    if line.strip() != '':
        if line.strip()[0] == '!' and not line.lstrip().startswith('!$OMP'):
            statement = line


    return statement,original_lines

#end assemble_continued_statement

#==================================================================



def remove_trailing_comment(line):

    if '!' not in line:
        return line

    else: 
        if len(match_single_quote.findall(line.split('!')[0]))%2 == 0 and len(match_double_quote.findall(line.split('!')[0]))%2 == 0:
            return line.split('!')[0]
        else:
            return line



#==================================================================



def remove_strings_from_line(line, owning_routine='',filename=''):  # and also remove a trailing comment if there is one

    destrung_line = ''

    ignore_starts = ['!','#']
    if filename.endswith('.F') : # this is F77 code, so 'C' is a comment ...
        ignore_starts.append('C')

    if any( line.lstrip().startswith(this) for this in ignore_starts):
        return line # these lines are sage to ignore

    elif len(match_single_quote.findall(line)) == 0 and len(match_double_quote.findall(line)) == 0: # no quotes at all in this line
        return line.split('!')[0]

#    elif '"' not in line: # there are *only* single quotes in this line
#        temp_line = line
#        while "'" in temp_line and ('!' not in temp_line or temp_line.find("'") < temp_line.find('!') ): # advance by pairs
#            destrung_line = destrung_line + temp_line[:temp_line.find("'")]
#            temp_line = temp_line[temp_line.find("'",temp_line.find("'")+1 ):] # temp_line from the second ' onwards
#        destrung_line = destrung_line + temp_line.split('!')[0]
#    elif "'" not in line: # there are *only* double quotes in this line
#        temp_line = line
#        while '"' in temp_line and ('!' not in temp_line or temp_line.find('"') < temp_line.find('!') ): # advance by pairs
#            destrung_line = destrung_line + temp_line[:temp_line.find("'")]
#            temp_line = temp_line[temp_line.find("'",temp_line.find("'")+1 ):] # temp_line from the second ' onwards
#        destrung_line = destrung_line + temp_line
    else : # there can be a combination of single and double quotes in this line
        temp_line = line
        #while len(match_any_quote.findall(temp_line.split('!')[0]))>0 : # advance by pairs
        num_its = 0
        while len(match_any_quote.findall(temp_line))>0 : # advance by pairs
            next_single = temp_line.find("'")
            next_double = temp_line.find('"')
            next_exclam = temp_line.find('!')
            if next_single == -1 : next_single = 9999
            if next_double == -1 : next_double = 9999
            if next_exclam == -1 : next_exclam = 9999
            #print (next_single, next_double,temp_line)
            #try:
            #    print (temp_line.rstrip(), next_single, next_double)
            #except UnicodeEncodeError:
            #    print(owning_routine)
            #    sys.exit()
            num_its += 1 
            if num_its > 200:
                print ('unfinishing loop : ',owning_routine, line)
                #line
                print(destrung_line,'...',temp_line)
                temp_line
                sys.exit()
            if   next_single < next_double and next_single <= next_exclam:
                next_quote = "'"
                nqp = next_single  # next quote position
            elif next_double < next_single and next_double <= next_exclam:
                next_quote = '"'
                nqp = next_double 
            elif next_single == next_double or next_exclam < min(next_single,next_double): # they have the same value, i.e. there is neither ' nor "
                return destrung_line + temp_line.split('!')[0]
            destrung_line = destrung_line + temp_line[:nqp]
            temp_line = temp_line[temp_line.find(next_quote,nqp+1 )+1:] # temp_line from the second ' onwards
        destrung_line = destrung_line + temp_line.split('!')[0]
    

    return destrung_line

#end remove_strings_from_line

#======================================================


def register_main_program(filename):
    """
    the main program must be put into a separate group
    """
    
    ## set group of current file to MAIN in clean_source_files2

    if clean_source_files2[filename][2] != "MAIN":
        t = clean_source_files2[filename]
        clean_source_files2[filename] = (t[0],t[1],"MAIN",t[3],t[4],t[5],t[6])

    

#end register_main_program

#======================================================

# function which returns the first argument in a string of arguments, and the rest of the (shortened by one arg) input string
def glob_arg(arg_string, full_line):
    first_arg = ''
    rest_of_line = ''
    open_brackets = 0
    in_first_arg = True


    for next_char in arg_string:
        if next_char == ',' and open_brackets == 0:
            in_first_arg = False 

        elif next_char == '(' :
            open_brackets = open_brackets + 1
        elif next_char == ')' :
            open_brackets = open_brackets - 1
            if open_brackets < 0 :
                raise SystemError(' we should not have this brackets situation :   ',arg_string, '...', full_line )
            
        if in_first_arg : 
            first_arg = first_arg + next_char
        else:
            rest_of_line = rest_of_line + next_char
        
    rest_of_line = rest_of_line.strip() # make sure we have no pesky leading or trailing white spaces

    if rest_of_line != '':
        first_arg = first_arg+rest_of_line[0] # append the comma 
        rest_of_line = rest_of_line[1:] # get rid of leading comma

    return first_arg,rest_of_line

#end glob_arg


#===============================================


def uppercase_except_strings(line,routine_name) :
    
    return_line = ''

    pr = False
    #if 'SUJQCOR(NFLEVG' in line:
    #    print('going to look at line:\n')#,line)
    #    pr = True
    
    # the easy case : no string in the line
    if line.strip().upper().startswith('!$OMP'):
        return line.upper()
    elif line.strip().startswith('#'):
        return pretty_preprocess(line)
    elif '"' not in line and "'" not in line:
        if '!' in line:
            return line.split('!')[0].upper() + line[line.find('!'):]
        else:
            return line.upper()

    elif line.lstrip().startswith('!'):
        return line

#    elif len(re.findall('"',line.split('!')[0]))%2==1 or len(re.findall("'",line.split('!')[0]))%2==1:
#        print('check this case !', line) 
#        sys.exit()


    else: # we have a positive number of either '"' or "'"
        temp_line = line
        thru_loops = 0
        while temp_line != '':
            #print ('in while: ',temp_line, return_line)
            if pr : print ('in while: ',return_line,'...',temp_line,'...')
            thru_loops += 1

            first_single = temp_line.find("'")
            first_double = temp_line.find('"')
            exclamation = temp_line.find('!')
            if first_single == -1 :
                quote = '"' # we're looking for "
                first_quote_pos = first_double
            elif first_double == -1 :
                quote = "'"
                first_quote_pos = first_single
            elif first_single < first_double:
                quote = "'"

                first_quote_pos = first_single
            elif first_single > first_double:
                quote = '"'
                first_quote_pos = first_double
            else:
                raise SystemError('f__ing up in uppercaser: ', line)

            if pr:
                print (first_single,first_double,first_quote_pos,exclamation,'...',temp_line,'...',return_line)
                
            while exclamation != -1 and exclamation < first_quote_pos:
                #if '\n' in (temp_line.split('!')[1])[0:len(temp_line.split('!')[1])-1]: # comment with a continued line after
                if '\n' in temp_line[temp_line.find('!'):-1]: # comment with a continued line after
                    return_line = return_line + temp_line.split('!')[0].upper()  + (temp_line[temp_line.find('!'):]).split('\n')[0]+'\n'

                    if pr: print ('passing 1: ',temp_line)
                    temp_line = temp_line[temp_line.find('!'):] ## .split('!')[1]
                    temp_line = temp_line[ temp_line.find('\n')+1:]
                    if pr: print ('1 bis: ',temp_line)
                else:
                    return_line = return_line + temp_line.split('!')[0].upper() + temp_line[temp_line.find('!'):]
                    if pr: print ('passing 2')
                    temp_line = ''
                first_single = temp_line.find("'")
                first_double = temp_line.find('"')
                exclamation = temp_line.find('!')
                if first_single == -1 :
                    quote = '"' # we're looking for "
                    first_quote_pos = first_double
                elif first_double == -1 :
                    quote = "'"
                    first_quote_pos = first_single
                elif first_single < first_double:
                    quote = "'"
                    first_quote_pos = first_single
                elif first_single > first_double:
                    quote = '"'
                    first_quote_pos = first_double
                #else:
                #    raise SystemError('f__ing up in uppercaser: ', line)


            if pr:
                print ('after excl: ',first_single,first_double,first_quote_pos,exclamation,temp_line,'...',return_line)
                input('======')

            if thru_loops > 150 :
                print('check: ', first_single,first_double,first_quote_pos, exclamation,temp_line, '...',return_line)
                input('...')

            if temp_line != '':
                if first_single == -1 and first_double == -1: # the end of the string no longer has any quotes
                    if '!' not in temp_line:
                        return_line = return_line + temp_line.upper()
                        temp_line = ''
                    else:
                        #if '\n' in (temp_line.split('!')[1])[0:len(temp_line.split('!')[1])-1]:
                        if '\n' in temp_line[temp_line.find('!'):-1]:
                            return_line = return_line + temp_line.split('!')[0].upper()  + (temp_line[temp_line.find('!'):]).split('\n')[0]+'\n'
                            temp_line = temp_line[temp_line.find('!'):] ## .split('!')[1]
                            temp_line = temp_line[ temp_line.find('\n')+1:]                            
                        else:
                            return_line = return_line + temp_line.split('!')[0].upper() + temp_line[temp_line.find('!'):]
                            temp_line = ''
                else:
                    return_line = return_line + temp_line[:first_quote_pos].upper() + temp_line[first_quote_pos:temp_line.find(quote,first_quote_pos+1)+1] 
                    if temp_line[-1] == '\n' and temp_line.find(quote,first_quote_pos+1) == len(temp_line)-2 : # the quote is the last character of the line ...
                        return_line = return_line + '\n'
                        temp_line = ''
                    elif temp_line[-1] != '\n' and temp_line.find(quote,first_quote_pos+1) == len(temp_line)-1 : # the quote is the last character of the line ...
                        return_line = return_line + '\n'
                        temp_line = ''
                    else: 
                        if temp_line.find(quote,first_quote_pos+1) == -1: raise SystemError('this is the problem? searching for '+quote+' in '+
                                                                                            temp_line+'...\n'+line+'...'+routine_name)
                        temp_line = temp_line[temp_line.find(quote,first_quote_pos+1)+1:] 

                #        return_line = line[:line.find(quote)+1].upper()+line[line.find(quote)+1:line.rfind(quote)] + line[line.rfind(quote):].upper()


        if pr : 
            print ('after while: ',return_line,'...',temp_line,return_line[-1]=='\n')
            input('...')
        return return_line
#end uppercase_excpet_strings



#================================================


def remove_strings(line) :  # superseded by remove_strings_from_line
    
    ret_line = ''

    pr = False
    #if '! bug found by Jason 01/2008' in line:
    #    print('going to look at line:\n')#,line)
    #    pr = True
    
    # the easy case : no string in the line
    if line.strip().upper().startswith('!$OMP'):
        return line
    elif line.strip().startswith('#'):
        return pretty_preprocess(line)
    elif '"' not in line and "'" not in line:
        return line

    elif line.lstrip().startswith('!'):
        return line


    else: # we have a positive number of either '"' or "'"
        temp_line = line
        thru_loops = 0
        while temp_line != '':
            #print ('in while: ',temp_line, ret_line)
            if pr : print ('in while: ',ret_line,'...',temp_line,'...')
            thru_loops += 1

            first_single = temp_line.find("'")
            first_double = temp_line.find('"')
            exclamation = temp_line.find('!')
            if first_single == -1 :
                quote = '"' # we're looking for "
                first_quote_pos = first_double
            elif first_double == -1 :
                quote = "'"
                first_quote_pos = first_single
            elif first_single < first_double:
                quote = "'"
                first_quote_pos = first_single
            elif first_single > first_double:
                quote = '"'
                first_quote_pos = first_double
            else:
                raise SystemError('f__ing up in string remover: ', line)

            if pr:
                print (first_single,first_double,first_quote_pos,exclamation,'...',temp_line,'...',ret_line)
                
            while exclamation != -1 and exclamation < first_quote_pos:
                #if '\n' in (temp_line.split('!')[1])[0:len(temp_line.split('!')[1])-1]: # comment with a continued line after
                if '\n' in temp_line[temp_line.find('!'):-1]: # comment with a continued line after
                    ret_line = ret_line + temp_line.split('!')[0].upper()  + (temp_line[temp_line.find('!'):]).split('\n')[0]+'\n'

                    if pr: print ('passing 1: ',temp_line)
                    temp_line = temp_line[temp_line.find('!'):] ## .split('!')[1]
                    temp_line = temp_line[ temp_line.find('\n')+1:]
                    if pr: print ('1 bis: ',temp_line)
                else:
                    ret_line = ret_line + temp_line.split('!')[0].upper() + temp_line[temp_line.find('!'):]
                    if pr: print ('passing 2')
                    temp_line = ''
                first_single = temp_line.find("'")
                first_double = temp_line.find('"')
                exclamation = temp_line.find('!')
                if first_single == -1 :
                    quote = '"' # we're looking for "
                    first_quote_pos = first_double
                elif first_double == -1 :
                    quote = "'"
                    first_quote_pos = first_single
                elif first_single < first_double:
                    quote = "'"
                    first_quote_pos = first_single
                elif first_single > first_double:
                    quote = '"'
                    first_quote_pos = first_double
                #else:
                #    raise SystemError('f__ing up in uppercaser: ', line)


            if pr:
                print ('after excl: ',first_single,first_double,first_quote_pos,exclamation,temp_line,'...',ret_line)
                input('======')

            if thru_loops > 150 :
                print('check: ', first_single,first_double,first_quote_pos, exclamation,temp_line, '...',ret_line)
                input('...')

            if temp_line != '':
                if first_single == -1 and first_double == -1: # the end of the string no longer has any quotes
                    if '!' not in temp_line:
                        ret_line = ret_line + temp_line.upper()
                        temp_line = ''
                    else:
                        #if '\n' in (temp_line.split('!')[1])[0:len(temp_line.split('!')[1])-1]:
                        if '\n' in temp_line[temp_line.find('!'):-1]:
                            ret_line = ret_line + temp_line.split('!')[0].upper()  + (temp_line[temp_line.find('!'):]).split('\n')[0]+'\n'
                            temp_line = temp_line[temp_line.find('!'):] ## .split('!')[1]
                            temp_line = temp_line[ temp_line.find('\n')+1:]                            
                        else:
                            ret_line = ret_line + temp_line.split('!')[0].upper() + temp_line[temp_line.find('!'):]
                            temp_line = ''
                else:
                    ret_line = ret_line + temp_line[:first_quote_pos].upper() + temp_line[first_quote_pos:temp_line.find(quote,first_quote_pos+1)+1] 
                    if temp_line.find(quote,first_quote_pos+1) == len(temp_line)-2 : # the quote is the last character of the line ...
                        ret_line = ret_line + '\n'
                        temp_line = ''
                    else: 
                        if temp_line.find(quote,first_quote_pos+1) == -1: print('this is the problem? ',temp_line)
                        temp_line = temp_line[temp_line.find(quote,first_quote_pos+1)+1:] 

                #        ret_line = line[:line.find(quote)+1].upper()+line[line.find(quote)+1:line.rfind(quote)] + line[line.rfind(quote):].upper()


        if pr : 
            print ('after while: ',ret_line,'...',temp_line)
            input('...')
        return ret_line
#end remove_strings
               
#==================================

                
def match_string_case(string_to_match, string_to_add):
    
    output = string_to_add
    if string_to_match.islower(): # return in lower case
        return output.lower()
    elif string_to_match.isupper(): # return in upper case
        return output.upper()
    else:  # mixture of cases, but just return upper case for the moment
        return output.upper()
#end match_string_case


#==================================

def match_do_loop(words,line):
    do_index_var = ""
    if words[0] == 'DO':
        do_index_var = words[1]
    elif len(words) >= 2 :
        if words[1] == 'DO' :  # this should be a named list; maybe check?
            do_index_var = words[2]
            print('we seem to have a named do : \n',line,'\n')

    return do_index_var

#==================================

def match_end_do_loop(words):
    enddo_var = ''
    if words[0] == 'ENDDO':
        enddo_var = 'ENDDO'
    elif len(words) > 1:
        if words[0] == 'END' and words[1] == 'DO' :
            enddo_var = 'END DO'
    return enddo_var

#==================================

def count_nested_loops_depth(filename) :
    
    srclines = source_code_dict[filename][0]

    long_line = ""      # the line that will be sent to the splitter
    nest_level = 0
    max_nest_level = 0
    num_hor_loops = 0
    nested_indices_list = []

    pr = False
    if 'cloudsc.F90' in filename: pr = True  # switch on debugging printing
    if pr: print('looking at ',filename,' in count_loops')

    for line_num,fline in enumerate(srclines):

        ## the scanner. feed each line into the joiners

        long_line, get_next = join_and_strip_free(fline, long_line)


        if get_next:
            continue  ## break out, back to loop for with next fline

        #if pr: 
        #    print('\nlong line : \n', long_line)
        #    input('...')

        long_line = strip_strings(long_line)

        ## the splitter. split each line at ;
        ## this gives us one statement per line
        # incidentally, THIS SHOULD NOT BE COMMON IN THE IFS

        for line in long_line.split("$$$$$$"):
            line = line.strip()
            if line == "":
                continue
            if line.lstrip().startswith('!'):
                continue
            #lines += 1

            line = line.upper()  ## shift all characters to UPPERCASE

            fortran_words = re.split('\W+',line)

            # check if line starts a "do" loop
            name = match_do_loop(fortran_words,line)
            if name != "":
                nest_level += 1
                nested_indices_list.append(name)
                if nest_level > max_nest_level :
                    max_nest_level = nest_level # keep a record of the maximum nest level met in routine
                    list_at_max_level = copy.deepcopy(nested_indices_list)
                loop_var = identify_horiz_loop(fortran_words,line,filename)
                if loop_var != '' :
                    num_hor_loops += 1
                continue

            
            # check if line ends a "do" loop
            name = match_end_do_loop(fortran_words)
            if name != "":
                nest_level -= 1
                if nest_level == 0:
                    nested_indices_list = []
                else :
                    del nested_indices_list[-1]
                continue

        ## line done, reset
        long_line=""

    if max_nest_level > 0:
        print('nested loops in ',filename,': ',list_at_max_level)
        print('horizontal loops in ',filename,' : ',num_hor_loops,'\n')
#end def count_nested_loops_depth(filename)

#==================================

def identify_horiz_loop_arrays(filename) :
    
    srclines = source_code_dict[filename][0]

    long_line = ""      # the line that will be sent to the splitter
    nest_level = 0
    max_nest_level = 0
    num_hor_loops = 0
    nested_indices_list = []
    in_hor_loop = False
    hor_depth = 0

    arrays_to_demote = []

    pr = False
    if 'cloudsc.F90' in filename: pr = True  # switch on debugging printing
    if pr: print('looking at ',filename,' in count_loops')

    for line_num,fline in enumerate(srclines):

        ## the scanner. feed each line into the joiners

        long_line, get_next = join_and_strip_free(fline, long_line)


        if get_next:
            continue  ## break out, back to loop for with next fline

        #if pr: 
        #    print('\nlong line : \n', long_line)
        #    input('...')

        long_line = strip_strings(long_line)

        ## the splitter. split each line at ;
        ## this gives us one statement per line
        # incidentally, THIS SHOULD NOT BE COMMON IN THE IFS

        for line in long_line.split("$$$$$$"):
            line = line.strip()
            if line == "":
                continue
            if line.startswith('!'):
                continue
            #lines += 1

            line = line.upper()  ## shift all characters to UPPERCASE

            fortran_words = re.split('\W+',line)

            # check if line starts a "do" loop
            name = match_do_loop(fortran_words,line) # return loop index name if this is a loop
            if name != "":
                nest_level += 1
                nested_indices_list.append(name)
                if nest_level > max_nest_level :
                    max_nest_level = nest_level # keep a record of the maximum nest level met in routine
                    list_at_max_level = copy.deepcopy(nested_indices_list)
                temp_loop_var = identify_horiz_loop(fortran_words,line,filename)
                #if temp_loop_var
                if temp_loop_var != '' :
                    hor_loop_var = copy.copy(temp_loop_var)
                    num_hor_loops += 1
                    if in_hor_loop == True:
                        print('shit, nested horiz loops? ',line,filename)
                        input('...')
                    else :
                        in_hor_loop = True
                    hor_depth = 1 # found a horizontal loop, now try to track where it ends 
                continue

            # check if line ends a "do" loop
            name = match_end_do_loop(fortran_words)
            if name != "":
                nest_level -= 1
                #if hor_depth > 0 : # decrement horizontal depth if we're inside a horiz loop 
                #    hor_depth -= 1
                #if hor_depth == 0:
                #    in_hor_loop = False
                if nested_indices_list[-1] == 'JL' :
                    in_hor_loop = False
                    hor_loop_var = ''
                    
                if nest_level == 0:
                    nested_indices_list = []
                else :
                    del nested_indices_list[-1]

                continue

            # if we're inside a horizontal loop, then this line needs to be analyzed
            ##if hor_depth > 0 : 
            if in_hor_loop : 
                arrays_with_hor_index = scan_line_for_hor_arrays(hor_loop_var, fortran_words,line,filename)
                if len(arrays_with_hor_index) != 0 : #at least one array with JL on this line
                    arrays_to_demote = arrays_to_demote + arrays_with_hor_index

        ## line done, reset
        long_line=""

    arrays_to_demote = list(set(arrays_to_demote))
    if max_nest_level > 0:
        print('nested loops in ',filename,': ',list_at_max_level)
        print('horizontal loops in ',filename,' : ',num_hor_loops,'\n')

    if pr:
        print('number of arrays that need demoting : \n',len(arrays_to_demote) )
    return arrays_to_demote

#end def identify_horiz_loop_arrays(filename)

#==================================

def identify_horiz_loop(words, line, filename):

    return_val = ''
    pr = False
    if 'cloudsc.F90' in filename : pr = True

    lwords = copy.copy(words)
    # if first or second word is not a DO, something has gone wrong ...
    if lwords[0] != 'DO' : # treat possible named loop by ignoring first word if not DO
        del lwords[0]
    if lwords[0] != 'DO' : # now first word really should be a DO!
        raise SystemError('problem in identify_horiz_loop : line not starting with DO \n' + line + filename+'\n\n')

    lline = copy.copy(line)
    lline = strip_comments(lline)
    if len(re.findall('=',lline)) != 1:
        print('identify_horiz_loop : why is there a strange number of = in this line?\n',line)
        input('...')
    lline = lline.split('=')[1].strip() # extract part of line after the '=' sign
    loop_var = lwords[1]
    lower_bound = lline.split(',')[0].strip()  # the first item before the first comma
    if lline.split(',')[1].strip()[0].isdigit(): # first char after comma is digit, so we have an integer as an upper bound
        obj = re.search(r'^[0-9]+',lline.split(',')[1].strip())
        upper_bound = obj.group()
        #print('integer upper bound for loop index here ... ', line)
        #input('...')
    else:
        upper_bound = re.split('\W+', lline.split(',')[1].strip())[0]  # first word after comma

    #if pr:
    #    print('checking loop identification : \n',loop_var,lower_bound, upper_bound,'\n',line,'\n\n')

        
    possible_bounds = ['KIDIA','KFDIA','1','NPROMA']
    if loop_var == 'JL' and lower_bound in possible_bounds and upper_bound in possible_bounds:
        #print('hurray, got one! ',loop_var,lower_bound, upper_bound,'\n',line,'\n\n')
        #input('...')
        return_val = copy.copy(loop_var)
    #else :
    #    print('why isnt loop_var found? ',loop_var, line)
    #    input('...')
    return return_val
#end def identify_horiz_loop(words, line, filename)

#==================================

def scan_line_for_hor_arrays(loop_var, words, line, filename):
    if loop_var == '' :
        print('why is loop_var empty?? ',line)
    # identify arrays from which an nproma loop is to be removed
    arrs_list = []
    if loop_var in words : # line contains our horizontal loop index of interest 
        lline = copy.copy(line)
        while bool( re.search( '\( *'+loop_var , lline )) == True :
            # careful, following doesn't identify derived type members as special
            start = re.split('\( *'+loop_var , lline )[0]
            arr_name = re.split('\W+',start)[-1]  # last word before '(JL'
            if len(start) > len(arr_name):
                if start[-1-len(arr_name)] == '%' :
                    arr_name = re.split('\W+',start)[-2]+'%'+arr_name
                    print('YEEEY, got one ! ', start, '---',arr_name)
            if arr_name == '' :
                print(lline,'---',start,'---',re.split('\W+',start),'---',loop_var,':::')
                input('...')
            arrs_list.append(arr_name)
            lline = lline[len(start)+3 : ]
        #arrs_list = []
    if 'UMH1' in arrs_list:
        print('umh1 culprit : ',line, arrs_list) 
    return arrs_list 
#end def scan_line_for_hor_arrays(loop_var, words, line, filename)

#==================================

def find_arrays_in_declarations(decl_lines, list_of_arrays, filename):

    #decl_lines is a list of tuples, so extract list of lines
    local_decl_lines = [tup[0] for tup in decl_lines]

    local_arr_list = copy.copy(list_of_arrays) # we hope this copy becomes empty at end of routine
    local_arr_list = [arr.split('%')[0] for arr in local_arr_list]
    for lne in local_decl_lines :
        if lne.lstrip().startswith('!'):
            continue
        wds = re.split('\W+',lne)
        if any(arr in wds for arr in local_arr_list):
            #dupl_list = [el for el in local_arr_list if el in wds]
            temp = [el for el in local_arr_list if el not in wds]
            local_arr_list = copy.copy(temp)

    if local_arr_list == []:
        print('arrays to be demoted for ',filename,' all found in declarations!')
    else :
        print('didnt find all arrays to be demoted in declarations for ',filename)
        print('missing : \n',local_arr_list)

#end def find_arrays_in_declarations

#==================================

def generate_single_column_code(filename, list_of_arrays):
    
    src_lines = source_code_dict[filename][0]
    src_iter = iter(src_lines)

    local_arr_list = copy.copy(list_of_arrays) # we hope this copy becomes empty at end of routine
    local_arr_list = [arr.split('%')[-1] for arr in local_arr_list]

    check_list = copy.copy(list_of_arrays) # use this list to check if there are any changes needed in a code line
    check_list = list(set( [arr.split('%')[0] for arr in list_of_arrays] ))
    
    
    # the arrays to add are the arrays being used from inside derived types
    ##arrays_to_add = []
    ##for el in list_of_arrays:
    ##    if '%' in el: # this is a derived type array
    ##        arrays_to_add.append( el.split('%')[0] + '_' + el.split('%')[1] )
    ##print('arrays to be added : \n',arrays_to_add)
    ### derived types to remove : those whose arrays we're now adding
    ##dtypes_to_remove = list(set( [el.split('%')[0] for el in list_of_arrays if '%' in el] ))
    ##print('derived types to be removed : \n',dtypes_to_remove)
    dtype_arrays = [el for el in list_of_arrays if '%' in el]
    derived_types_list = list(set( [el.split('%')[0] for el in dtype_arrays if '%' in el] ))
    derived_types_dict = {}
    for el in derived_types_list:
        derived_types_dict[el] = list(set( [el.split('%')[1] for el in dtype_arrays if '%' in el] ))
    print('dictionary of dtype_arrays for routine : \n', derived_types_dict)

    routine_name = source_code_dict[filename][1]

    position_in_file = 'before_routine'
    routine_code_lines = []
    routine_decl_lines = []
    found_an_assoc = False
    nest_level = 0
    num_hor_loops = 0
    in_hor_loop = False
    nested_indices_list = []
    declaration_line_start = '' # 'REAL(KIND=JPRB), INTENT('+intent+') :: '
    long_line = ''
    prev_long_line = ''
    now_scalars = []

    new_filename = filename + '_1cv'
    new_file = open(new_filename,'w', encoding="latin-1", errors="surrogateescape") 

    new_file.write('MODULE '+routine_name+'_1c_mod\n\n')
    new_file.write('CONTAINS\n\n')
    
    ## read in joined line by joined line and attribute lines to the right variable
    for line in (src_iter):
        prev_long_line = copy.copy(long_line)
        ## join lines that should be joined
        #long_line,orig_lines = assemble_continued_statement(line, src_lines)
        long_line,orig_lines = assemble_continued_statement_from_iterator(line, src_iter)

        ## split joined line into words
        words = re.split('\W+',remove_strings_from_line( long_line.lstrip().upper() ) )

        ## have we found the routine?
        if position_in_file=='before_routine' and not long_line.lstrip().startswith('!' ) and \
           (words[0] in ['SUBROUTINE','FUNCTION'] and words[1] == routine_name  or \
            (len(words)>2 and all( el in words for el in ['FUNCTION' , routine_name]) and words[0] != 'END' ) ):
            print('Found routine in file, now looking at comments!')
            position_in_file = 'in_comments'
            new_file.write( adjust_decl(long_line, orig_lines, derived_types_dict ) )
            continue

        ## deal with the comments zone
        if position_in_file=='in_comments':
            if long_line.lstrip()=='' :
                new_file.write( ''.join(orig_lines) )
                #long_line = ''
                continue
            elif long_line.lstrip()[0]=='!':
                #print('comment line : ',long_line)
                new_file.write( ''.join(orig_lines) )
                #long_line = ''                    
                continue
            else:  ## we've got to the end of the comments 
                position_in_file = 'in_MODULES'
                #routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                #long_line = ''
                #continue
                print('Finished routine comments, now looking at modules')
                new_file.write( ''.join(orig_lines) ) # possibly at some point we should check whether to remove derived type usage
                continue


        ## deal with the USE MODULES
        if position_in_file=='in_MODULES':
            if words[0]=='USE' or line.lstrip()=='' :
                #routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                #print('USE line : ',long_line)
                new_file.write( ''.join(orig_lines) ) # possibly at some point we should check whether to remove derived type usage
                long_line=''
                continue
            elif long_line.lstrip().startswith('!') : # dont really care about this line for the moment
                routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                #print('USE line : ',long_line)
                new_file.write( ''.join(orig_lines) ) # possibly at some point we should check whether to remove derived type usage
                long_line=''
                continue
            elif words[0] == 'IMPLICIT' : # use this to switch explicitly to 'in_declarations'
                position_in_file = 'in_declarations'
                #routine_decl_lines.append( (long_line, orig_lines) ) ## debatable as to whether this should be put in or not
                #print('DECL line : ',long_line)
                #new_file.write( ''.join(orig_lines) ) 
                new_file.write( '\n'.join(orig_lines) )
                
                long_line=''
                print('Finished routine modules, now looking at declarations')
                continue

            else:  ## we've got to the argument declarations
                #routine_argument_specs_list.append(long_line)
                position_in_file == 'in_declarations'
                routine_decl_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                new_list_lines = copy.copy(orig_lines)
                for arr in [el in words for el in local_arr_list] :
                    #temp_lines = [remove_hor_from_array(arr,line)[0] for line in new_list_lines] # problem with tuple, try running on ''.join(origlines)
                    temp_lines,list_of_scal = remove_hor_from_array(arr,''.join(new_list_lines)) 
                    new_list_lines = copy.copy(temp_lines)
                    if list_of_scal != [] and 'INTENT' in strip_comments(orig_lines[0]): 
                        now_scalars += list_of_scal
                if any( [dtype in words for dtype in derived_types_list] ) :
                    continue
                new_file.write( ''.join(new_list_lines) ) 
                #long_line = ''
                continue


        ## deal with the variable declarations, header includes, and ASSOCIATE
        if position_in_file=='in_declarations':
            if words[0]=='IF' and 'LHOOK' in words: ## this is the end of declarations
                #line=''
                position_in_file = 'routine_ASSOCIATE'
                new_file.write( ''.join(orig_lines) )
                print('Finished drhook call, now looking at ASSOCIATE')
                #long_line = ''
                continue
            elif long_line.startswith('#include') or long_line.lstrip().startswith('!') or long_line.strip()=='':
                #routine_decl_lines.append( (long_line, orig_lines) ) # dont actually need to store this line
                new_file.write( ''.join(orig_lines) )
                #long_line = ''
                continue                    
            elif not long_line.lstrip().startswith('!') and words[0] not in decl_keywords : 
                print("we don't seem to have a DrHook call for this routine : ", filename, '*'+long_line+'*')
                input('...')
                position_in_file = 'routine_ASSOCIATE'
                #long_line = ''
                continue
            elif line.strip() == '' : 
                #routine_decl_lines.append( (long_line, orig_lines) ) dont need to store
                new_file.write( ''.join(orig_lines) )
                #long_line = ''
                continue
            else:  ## we're still in the argument declarations
                #routine_decl_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                routine_decl_lines.append( (strip_comments(long_line), orig_lines) ) ## added to check declarations for arg intents
                #new_file.write( ''.join(orig_lines) )
                new_list_lines = copy.copy(orig_lines)
                #input('got an array? '+''.join(orig_lines))
                for arr in [el for el in  words if el in local_arr_list] :
                    print('calling remove_hor with ',arr,' on ', new_list_lines[0])
                    #temp_lines = [remove_hor_from_array(arr,line) for line in new_list_lines] # tuple output of remove_hor is a problem here
                    temp_lines,list_of_scal = remove_hor_from_array(arr,''.join(new_list_lines)) 
                    if list_of_scal != [] and 'INTENT' in strip_comments(orig_lines[0]): 
                        now_scalars += list_of_scal
                    new_list_lines = copy.copy(temp_lines)
                if any( [dtype in words for dtype in derived_types_list] ) :
                    intent = words[ words.index('INTENT') + 1 ]
                    if intent not in ['IN','OUT','INOUT']:
                        raise SystemError('problem with dtype declaration : '+''.join(lines_list))
                    if declaration_line_start == '':
                        declaration_line_start = ' '*(len(prev_long_line)-len(prev_long_line.lstrip())) + 'REAL(KIND=JPRB)'
                        if ',' not in prev_long_line :
                            print('strange case here : ',prev_long_line,'---\n',long_line)
                            input('...')
                        comma_pos = prev_long_line.index(',')
                        declaration_line_start += ' '*(comma_pos-len(declaration_line_start)) + ','
                        intent_pos = len(prev_long_line.split('INTENT')[0])
                        declaration_line_start += ' '*(intent_pos-len(declaration_line_start)) + 'INTENT('+intent+')'
                        colon_pos = prev_long_line.index(':')
                        declaration_line_start += ' '*(colon_pos-len(declaration_line_start)) + ':: '
                    new_list_lines = declare_dtype_arrays(new_list_lines, words, derived_types_list,derived_types_dict,declaration_line_start)
                    declaration_line_start = ''
                    #continue
                new_file.write( ''.join(new_list_lines) ) 
                #long_line = ''
                continue

        ## deal with ASSOCIATE
        if position_in_file=='routine_ASSOCIATE':
            if long_line.strip() == '' :
                new_file.write( ''.join( orig_lines ) )
                #long_line = ''
                continue
            elif words[0] != 'ASSOCIATE':
                
                if found_an_assoc == False : 
                    print('We seem to have no ASSOCIATE for '+routine_name+'\n')

                position_in_file = 'in_routine'
                routine_code_lines.append( (long_line, orig_lines) )
                new_file.write( ''.join( orig_lines ) )
                new_file.write('\n')
                claw_lines = '!$claw define dimension jl(1:klon) &\n!$claw parallelize &\n!$claw scalar('
                claw_lines += ','.join( now_scalars ) + ')\n'
                new_file.write( claw_lines )
                new_file.write('\n')
                #long_line = ''
                continue
            else:

                routine_code_lines.append( (long_line, orig_lines) ) ## added to check declarations for arg intents
                #position_in_file = 'in_routine'
                new_file.write( ''.join( orig_lines ) )
                if found_an_assoc == False:
                    print('Finished an ASSOCIATE statement, now either one more, or routine code ...')
                    found_an_assoc = True
                else :
                    print('Finished 2nd ASSOCIATE statement, now either a third one (really?), or routine code ...')
                continue


        if position_in_file=='in_routine' and words[0]=='END' and words[1] in ['SUBROUTINE','FUNCTION'] and \
           (len(words)==2 or words[2]==routine_name):
            routine_code_lines.append( line )
            new_file.write( ''.join( orig_lines ) )
            position_in_file = 'after_routine'
            print('Found end of subroutine')
            #long_line = ''
            continue
            

        # now we are in routine, so have to remove horizontal loops, and remove horizontal indexes from arrays
            
        #if not any( arr in words for arr in local_arr_list) or long_line.lstrip().startswith('!'):
        #    new_file.write( '\n'.join(orig_lines) )
        #    continue
        #else : # got to do something with this line
        #    if position_in_file == 'in_declarations' :
        #        new_lines = copy.copy(orig_lines)
        #        #for arr in [el in words for el in local_arr_list] :
        #        #    temp_lines = [remove_hor_from_array(arr,line) for line in new_lines]
        #        #    new_lines = copy.copy(temp_lines)
        #        new_file.write( '\n'.join(new_lines) )
        
        name = match_do_loop(words,long_line) # return loop index name if this is a loop
        if name != "":
            nest_level += 1
            nested_indices_list.append(name)
            temp_loop_var = identify_horiz_loop(words,long_line,filename)
            if temp_loop_var != '' :
                hor_loop_var = copy.copy(temp_loop_var)
                num_hor_loops += 1
                if in_hor_loop == True:
                    print('shit, nested horiz loops? ',line,filename)
                    input('...')
                else :
                    in_hor_loop = True
                hor_depth = 1 # found a horizontal loop, now try to track where it ends 
                # this is a horizontal loop line, so we do *not* want it; write it out commented
                new_file.write( '!!'+''.join(orig_lines) )
                continue
            else :
                new_file.write( ''.join(orig_lines) )
                continue

        # check if line ends a "do" loop
        name = match_end_do_loop(words)
        if name != "":
            nest_level -= 1
            #if hor_depth > 0 : # decrement horizontal depth if we're inside a horiz loop 
            #    hor_depth -= 1
            #if hor_depth == 0:
            #    in_hor_loop = False
            line_start=''
            if nested_indices_list[-1] == 'JL':
                if in_hor_loop == False:
                    print('why are these not true together ?')
                    print(long_line)
                    input(';;;')
                in_hor_loop = False
                hor_loop_var = ''
                line_start = '!!' # we want to comment out this END DO which closes a horizontal loop

            if nest_level == 0:
                nested_indices_list = []
            else :
                del nested_indices_list[-1]
            new_file.write( line_start + ''.join(orig_lines) )

            continue
        
        # commented lines can be kept as is
        if orig_lines[0].lstrip().startswith('!') or ''.join(orig_lines).strip() == '':
            new_file.write( ''.join(orig_lines) )
            continue

        # now deal with lines that have an array or dtype_array needing to be demoted
        if any(wd in words for wd in check_list) :
            # this line needs work
            #nline, new_scalars = adjust_code_line(orig_lines, words, check_list, list_of_arrays)
            nline = adjust_code_line(orig_lines, words, check_list, list_of_arrays)
            new_file.write( nline )

    new_file.write('END MODULE '+routine_name+'_1c_mod')
    new_file.close()

    print('scalar arrays : ', list(set(now_scalars)) )

#end def generate_single_column_code(filename, list_of_arrays)

#==================================
# hopefully does what the name suggests
def remove_associates(filename):
    """
    Remove ASSOCIATE statement and replace all association mappings.
    """

    # Parse the given file into our internal format
    from ecir.sourcefile import FortranSourceFile
    newfile = FortranSourceFile('%s_1cv' % filename)

    # Pick ASSOCIATE statement from longlines
    assoc_lines = [l for l in newfile.longlines if 'ASSOCIATE' in l]
    assert len(assoc_lines) == 1
    assoc_line = assoc_lines[0]
    assert('ASSOCIATE' in assoc_line)

    # Extract association map from statement
    content = assoc_line.split('ASSOCIATE')[1].strip(' ()\n')
    raw_pairs = [s.strip(' &\n') for s in content.split(',')]
    associations = dict([tuple(p.split('=>')) for p in raw_pairs])
    # TODO: associations should probably be sanity-checked

    # Strip original ASSOCIATE statement
    re_strip_assoc = re.compile('ASSOCIATE\([^)]*\)\n', re.MULTILINE)
    newfile._raw_string = re_strip_assoc.sub(repl='', string=newfile._raw_string)

    # Replace associations in the subroutine body
    newfile.replace(associations)
    
    # Write out the generated file
    newfile.write(filename='%s_assoc2' % newfile.name)

    new_filename = filename + '_1cv'
    out_filename = copy.copy(new_filename)
    
    try:
        out = (open(new_filename,'rt', encoding="ascii", errors="surrogateescape")).read()
        out_filename = new_filename + '_assoc'
    except IOError:
        print ("single column version not yet written, so working on original") #Does not exist OR no read permissions
        out = (open(filename,'rt', encoding="ascii", errors="surrogateescape")).read()
        out_filename = filename + '_assoc'

    src_lines = [l+'\n' for l in out.split('\n')]
    src_iter = iter(src_lines)

    routine_name = source_code_dict[filename][1]

    position_in_file = 'before_routine'
    routine_code_lines = []
    routine_decl_lines = []
    found_an_assoc = False
    nest_level = 0
    num_hor_loops = 0
    in_hor_loop = False
    nested_indices_list = []
    declaration_line_start = '' # 'REAL(KIND=JPRB), INTENT('+intent+') :: '
    long_line = ''
    prev_long_line = ''
    now_scalars = []

    previous_associations = []

    new_file = open(out_filename,'w', encoding="latin-1", errors="surrogateescape") 

    ## read in joined line by joined line and attribute lines to the right variable
    for line in (src_iter):
        prev_long_line = copy.copy(long_line)
        ## join lines that should be joined
        #long_line,orig_lines = assemble_continued_statement(line, src_lines)
        long_line,orig_lines = assemble_continued_statement_from_iterator(line, src_iter)

        ## split joined line into words
        words = re.split('\W+',remove_strings_from_line( long_line.lstrip().upper() ) )

        ## have we found the routine?
        if position_in_file=='before_routine' and not long_line.lstrip().startswith('!' ) and \
           (words[0] in ['SUBROUTINE','FUNCTION'] and words[1] == routine_name  or \
            (len(words)>2 and all( el in words for el in ['FUNCTION' , routine_name]) and words[0] != 'END' ) ):
            position_in_file = 'in_comments'
            new_file.write( ''.join(orig_lines) )
            continue

        ## deal with the comments zone
        if position_in_file=='in_comments':
            if long_line.lstrip()=='' :
                new_file.write( ''.join(orig_lines) )
                continue
            elif long_line.lstrip()[0]=='!':
                new_file.write( ''.join(orig_lines) )
                continue
            else:  ## we've got to the end of the comments 
                position_in_file = 'in_MODULES'
                new_file.write( ''.join(orig_lines) ) # possibly at some point we should check whether to remove derived type usage
                continue


        ## deal with the USE MODULES
        if position_in_file=='in_MODULES':
            if words[0]=='USE' or line.lstrip()=='' :
                new_file.write( ''.join(orig_lines) ) 
                continue
            elif long_line.lstrip().startswith('!') : # dont really care about this line for the moment
                new_file.write( ''.join(orig_lines) ) 
                continue
            elif words[0] == 'IMPLICIT' : # use this to switch explicitly to 'in_declarations'
                position_in_file = 'in_declarations'
                new_file.write( '\n'.join(orig_lines) )                
                continue

            else:  ## we've got to the argument declarations
                position_in_file == 'in_declarations'
                new_file.write( ''.join(orig_lines) ) 
                continue


        ## deal with the variable declarations, header includes, and ASSOCIATE
        if position_in_file=='in_declarations':
            if words[0]=='IF' and 'LHOOK' in words: ## this is the end of declarations
                position_in_file = 'routine_ASSOCIATE'
                new_file.write( ''.join(orig_lines) )
                print('Finished drhook call, now looking at ASSOCIATE')
                continue
            elif long_line.startswith('#include') or long_line.lstrip().startswith('!') or long_line.strip()=='':
                new_file.write( ''.join(orig_lines) )
                continue                    
            elif not long_line.lstrip().startswith('!') and words[0] not in decl_keywords : 
                print("we don't seem to have a DrHook call for this routine : ", filename, '*'+long_line+'*')
                input('...')
                position_in_file = 'routine_ASSOCIATE'
                continue
            elif line.strip() == '' : 
                new_file.write( ''.join(orig_lines) )
                continue
            else:  ## we're still in the argument declarations                
                new_file.write( ''.join(orig_lines) )
                continue

        ## deal with ASSOCIATE
        if position_in_file=='routine_ASSOCIATE':
            if long_line.strip() == '' :
                new_file.write( ''.join( orig_lines ) )
                continue
            elif words[0] != 'ASSOCIATE':                
                if found_an_assoc == False : 
                    print('We seem to have no ASSOCIATE for '+routine_name+'\n')
                position_in_file = 'in_routine'
                new_file.write( ''.join( orig_lines ) )
                continue
            else:

                # this line is an ASSOCIATE statement, so crack it to determine substitutions to perform
                if '!' in ''.join(orig_lines) :
                    print('going to have to deal with this ...')
                    input(''.join(orig_lines))
                associations = crack_associates(''.join(orig_lines))
                new_file.write( '!!'+''.join( orig_lines ) )
                print('Finished an ASSOCIATE statement, now either one more, or routine code ...')
                found_an_assoc = True
                continue


        if position_in_file=='in_routine' and words[0]=='END' and words[1] in ['SUBROUTINE','FUNCTION'] and \
           (len(words)==2 or words[2]==routine_name):
            routine_code_lines.append( line )
            new_file.write( ''.join( orig_lines ) )
            position_in_file = 'after_routine'
            print('Found end of subroutine')
            #long_line = ''
            continue
            

        # now we are in routine, so we have to insert ASSOCIATE information inline
        

#end def remove_associates(filename)

#==================================

def crack_associates(assoc_line):
    """
    Extract a ``name->expr`` map for ASSOCIATEs.
    """
    assert('ASSOCIATE' in assoc_line)
    content = assoc_line.split('ASSOCIATE')[1].strip(' ()\n')
    raw_pairs = [s.strip(' &\n') for s in content.split(',')]
    associations = dict([tuple(p.split('=>')) for p in raw_pairs])
    # TODO: associations should probably be sanity -checked
    return associations
        
    
#==================================

# add arrays and remove derived types from routine declaration, as required for single column code
def adjust_decl(concat_line, orig_lines, dtype_dict) : 

    # dtype_arrays : list of derived_type%array elements which need to be dealt with


    new_line  = ''
    new_line = ''.join(orig_lines) 
    
    if len(orig_lines) > 1 :
        line_cont = ' ' * (len(orig_lines[-1]) - len(orig_lines[-1].lstrip()))
        line_cont = line_cont + '&' + ' ' * (len(orig_lines[-1].lstrip()) - len(orig_lines[-1].lstrip().lstrip('&')) )
        #print( ' check line_cont :---'+line_cont+'---')
        #input('===')
    else :
        print('more work to do ...')
        raise SystemError('need to determine line_cont')
    
    for el in dtype_dict :
        #find the derived type in the long line, and replace it in situ by the necessary dtype_arrays
        arrs_to_add = [el+'_'+wd for wd in dtype_dict[el]]
        line_split = re.split(r'(?<=\b)'+re.escape(el)+r'(?=\b)', new_line)
        if len(line_split) != 2:
            print('unexpected split in routine declaration line : \n',new_line,line_split,'\n',el,'\n\n')
            input('...')
        line_start = line_split[0].rstrip('\n &') + '&\n'
        line_middle = line_cont + ', '.join( sorted(set( arrs_to_add )) ) + ',&\n'
        line_end = line_cont + line_split[1].lstrip(' ,&\n')
        #print('check components: \nline_start=\n',line_start,'line_middle=\n',line_middle,'line_end=\n',line_end)
        temp_line = line_start + line_middle + line_end
        #print('check temporary line : \n',temp_line)
        #input('===')
        new_line = copy.copy(temp_line)
    return new_line

#end def adjust_decl(concat_line, orig_lines, to_add, to_remove)

#==================================

# add arrays and  derived types in routine line, as required for single column code
def adjust_code_line(orig_lines, words, elt_list, fullname_list) : 

    # elt_list : 
    ##print('fullname_list : ', fullname_list)

    new_line  = ''
    new_line = copy.copy( ''.join(orig_lines) )
    
    #for el in elt_list :
    for el in fullname_list :
        if el in words or ('%' in el and el in new_line):
            # got to do something
            
            line_split = re.split(r'(?<=\b)'+re.escape(el)+r'(?=\b)', new_line)
            changed_line = copy.copy(line_split[0]) 
            if len(line_split) == 1:
                print('got a weird case : ',new_line, '--',el, '--',line_split)
                input('...')
            # now remove first index in following brackets
            for seg in line_split[1:]:
                changed_line = changed_line.rstrip('\n')+el 
                if not seg.lstrip().startswith('(') :
                    #print('this should be a check like ALLOCATED, or an array operation ... ',el,'---',new_line)
                    #input('===')
                    changed_line = changed_line.rstrip('\n') + seg + '\n'
                    continue
                bra = seg.find('(')
                changed_line += seg[:bra+1]
                next_cket = seg[bra+1:].find(')')
                if ',' not in seg[bra+1:bra+1+next_cket] :
                    # this array had ONLY a horizontal dimension
                    changed_line = changed_line[:-1]
                    changed_line +=seg[bra+1+next_cket+1:]
                    changed_line = changed_line.rstrip('\n') + '\n'
                    continue
                next_comma = seg[bra+1:].find(',')
                if '(' in seg[bra+1:bra+1+next_comma] :
                    print('feck ... ',seg,'---',bra,next_comma,seg[bra+1:bra+1+next_comma])
                    input('###')
                changed_line += seg[bra+1+next_comma+1:]
                changed_line = changed_line.rstrip('\n') + '\n'
            # deal with cases that only had horizontal index, leaving () after removal
            if '()' in changed_line :
                changed_line = changed_line.replace('()','')
            # deal with derived type arrays
            if '%' in el:
                dt_arr = el.replace('%','_')
                changed_line = re.sub( r'\b'+el+r'\b',dt_arr, changed_line)
            #print('check this line split : ', el,'\n',line_split,'\n',changed_line)
            #input('...')
            new_line = copy.copy(changed_line)

        

    return new_line

#end def adjust_code_line

#==================================

def remove_hor_from_array(arr_name, line) :
    #we assume for the moment that the horizontal dimension to remove is ALWAYS the first ...
    
    changed_line = remove_strings_from_line(line)
    now_scalar = []

    # we have already checked that the array is in the long_line before the call to this function,
    # but it is not necessarily in this particular segment of the long_line
    words = re.split('\W+', line)
    if arr_name in words :
        # we have to do something ...
        line_split = re.split( r'(?<=\b)'+re.escape(arr_name)+r'(?=\b)', changed_line )
        if len(line_split) == 1:
            raise SystemError('should never hit this case ')
        else:
            changed_line = line_split[0] + arr_name
            print ('got ',len(line_split),' segments in the split' )
            #if 'ZLCOND1' == arr_name : input('###')
            for seg in line_split[1:]:
                if not seg.lstrip().startswith('(') :
                    print('bug here : ',seg,'---',arr_name,'---',line)
                    input('===')    
                bra = seg.find('(')
                changed_line += seg[:bra+1]
                next_cket = seg[bra+1:].find(')')
                if ',' not in seg[bra+1:bra+1+next_cket] :
                    # this array had ONLY a horizontal dimension
                    changed_line = changed_line[:-1]
                    changed_line +=seg[bra+1+next_cket+1:]
                    changed_line = changed_line.rstrip('\n') + '\n'
                    now_scalar.append(arr_name)
                    continue
                next_comma = seg[bra+1:].find(',')
                if '(' in seg[bra+1:bra+1+next_comma] :
                    print('feck ... ',seg,'---',bra,next_comma,seg[bra+1:bra+1+next_comma])
                    input('###')
                changed_line += seg[bra+1+next_comma+1:]
                changed_line = changed_line.rstrip('\n') + '\n'
        print('check this line split : ', arr_name,'\n',line_split,'\n',changed_line)
        #if 'ZLCOND1' == arr_name : input('###')
        #input('...')

    return changed_line, now_scalar

#end def remove_hor_from_array()

#==================================

def declare_dtype_arrays(lines_list, words, dtype_list, dtype_dict, declaration_line_start) :
    # remove declaration of derived type in line, and add lines for array declarations
    if len(lines_list) == 1:
        lines_list = [ '!!'+lines_list[0] ]
    else:
        lines_list = [ '!!'+lines_list[0] , lines_list[1:] ]
    
    len_decl_start = len(declaration_line_start)
    for dt in [dtype for dtype in dtype_list if dtype in words]:
        arrs_to_add = [ dt+'_'+wd for wd in dtype_dict[dt] ]
        #print('arrs to add: ',arrs_to_add)
        new_line = declaration_line_start
        for arr in arrs_to_add:
            #new_line = new_line.rstrip('&\n')+', '+arr_name+
            if new_line.endswith('\n'):
                new_line = new_line + ' &'+' '*(len_decl_start-2)+arr+ '(KLEV), &\n' # ** here we are ASSUMING that the original dimensions are KLON,KLEV **
            else:
                new_line = new_line + arr + '(KLEV), &\n' # ** here we are ASSUMING that the original dimensions are KLON,KLEV **
        new_line = new_line.rstrip(', &\n')+'\n'
        lines_list.append( new_line )
    return lines_list
    
#==================================


                        

import toml
import sys


def transformToArgument(key, value, includeType = True, addUnderScore = False, includeOptional = False, functionOnly = False, typeFormat = 'pyBind', pyOnly = False):
    if not functionOnly and 'pythonArg' in value and value['pythonArg'] == False:
        return ""
    if not pyOnly and 'cppArg' in value and value['cppArg'] == True:
        return ""    
    if includeType:
        if typeFormat == 'pyBind':
            if 'tensor' in value['type']:
                type_str = f"torch::Tensor"
            else:
                type_str = f"{value['type']}"
            if 'optional' in value and value['optional']:
                type_str = f"std::optional<{type_str}>"
        elif typeFormat == 'compute':
            if 'tensor' in value['type']:
                ty = value['type'].split('[')[1].split(']')[0] if '[' in value['type'] else 'scalar_t'
                type_str = f"{'c' if 'const' not in value or value['const'] else ''}ptr_t<{ty}, {1 if 'dim' not in value else value['dim']}>"
            else:
                if value['type'] == 'double':
                    type_str = 'scalar_t'
                else:
                    type_str = f"{value['type']}"
            
    else:
        type_str = ""

    if addUnderScore:
        name_str = f"{key}_"
    else:
        name_str = f"{key}"

    if not includeOptional and 'optional' in value and value['optional']:
        return ""
    return f"{type_str} {name_str}"

def generateFunctionArguments(parsedToml, **kwargs):
    out = []

    for key, value in parsedToml.items():
        out.append(transformToArgument(key, value, **kwargs))

    out = [x for x in out if x != ""]



    return out
def generateTensorAccessors(parsedToml, optional = False):
    out = []
    for key, value in parsedToml.items():
        if 'cppArg' in value and value['cppArg'] == True:
            continue
        if optional: # only output optional values
            if 'optional' in value and value['optional']:
                if 'tensor' in value['type']:
                    ty = value['type'].split('[')[1].split(']')[0] if '[' in value['type'] else 'scalar_t'
                    dim = 1 if 'dim' not in value else value['dim']
                    out += [f"\tauto {key} = getAccessor<{ty}, {dim}>({key}_.value(), \"{key}\", useCuda, verbose_);\n"]
                else:
                    if value['type'] == 'double':
                        out += [f"\tauto {key} = (scalar_t) {key}_;\n"]
                    else:
                        out += [f"\tauto {key} = {key}_.value();\n"]
        else:
            if 'optional' in value and value['optional']:
                continue

            if 'tensor' in value['type']:
                ty = value['type'].split('[')[1].split(']')[0] if '[' in value['type'] else 'scalar_t'
                dim = 1 if 'dim' not in value else value['dim']

                out += [f"\tauto {key} = getAccessor<{ty}, {dim}>({key}_, \"{key}\", useCuda, verbose_);\n"]
            else:
                if value['type'] == 'double':
                    out += [f"\tauto {key} = (scalar_t) {key}_;\n"]
                else:
                    out += [f"\tauto {key} = {key}_;\n"]
    return out

def process(fileName):
    filePrefix = fileName.split('.')[0].split('/')[-1]

    with open(fileName, 'r') as f:
        lines = f.readlines()

        # print(lines)
    try:
        tomlBegin = lines.index('/** BEGIN TOML\n')
    except:
        print(f"Could not find Description in {fileName}, skipping file")
        return
    print(f"Generating bindings for {fileName} with prefix {filePrefix}")
    tomlBegin = lines.index('/** BEGIN TOML\n')
    tomlEnd = lines.index('*/ // END TOML\n')
    tomlDefinitions = ''.join(lines[tomlBegin + 1: tomlEnd])

    parsedToml = toml.loads(tomlDefinitions)


    endOfDefines = lines.index('/// End the definitions for auto generating the function arguments\n')

    prefixLines = lines[:tomlEnd+1]
    suffixLines = lines[endOfDefines:]

    if '// AUTO GENERATE ACCESSORS\n' in suffixLines:
        accessorBegin = suffixLines.index('// AUTO GENERATE ACCESSORS\n')
        accessorEnd = suffixLines.index('// END AUTO GENERATE ACCESSORS\n')
        accessorLines = generateTensorAccessors(parsedToml, optional = False)

        suffixLines = suffixLines[:accessorBegin+1] + accessorLines + suffixLines[accessorEnd:]

    if '// AUTO GENERATE OPTIONAL ACCESSORS\n' in suffixLines:
        accessorBegin = suffixLines.index('// AUTO GENERATE OPTIONAL ACCESSORS\n')
        accessorEnd = suffixLines.index('// END AUTO GENERATE OPTIONAL ACCESSORS\n')
        accessorLines = generateTensorAccessors(parsedToml, optional = True)

        suffixLines = suffixLines[:accessorBegin+1] + accessorLines + suffixLines[accessorEnd:]
        
    if '// GENERATE AUTO ACCESSORS\n' in suffixLines:
        accessorBegin = suffixLines.index('// GENERATE AUTO ACCESSORS\n')
        accessorEnd = suffixLines.index('// END GENERATE AUTO ACCESSORS\n')

        prefix = ['template<typename scalar_t = float>\n',
                  f'auto {filePrefix}_getFunctionArguments(bool useCuda, {filePrefix}_functionArguments_t){{\n']
        accessorLines = generateTensorAccessors(parsedToml, optional = False)
        suffix = [f'\treturn std::make_tuple({filePrefix}_arguments_t);\n',
                  '}\n']
        suffixLines = suffixLines[:accessorBegin+1] + prefix + accessorLines + suffix + suffixLines[accessorEnd:]

    numOptionals = len([x for x in parsedToml.values() if 'optional' in x and x['optional']])

    pyArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = True, functionOnly = False, typeFormat = 'pyBind', pyOnly = True))
    fnArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = False, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))
    computeArguments = ', '.join(generateFunctionArguments(parsedToml, includeOptional = False, functionOnly = True, typeFormat = 'compute', addUnderScore = False, pyOnly = False))
    arguments = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = False, functionOnly = True, typeFormat = 'pyBind', addUnderScore = False, pyOnly = False))
    arguments_ = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = False, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))

    if numOptionals > 0:
        fnArgumentsOptional = ', '.join(generateFunctionArguments(parsedToml, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))
        computeArgumentsOptional = ', '.join(generateFunctionArguments(parsedToml, includeOptional = True, functionOnly = True, typeFormat = 'compute', addUnderScore = False, pyOnly = False))
        argumentsOptional = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = False, pyOnly = False))
        argumentsOptional_ = ', '.join(generateFunctionArguments(parsedToml, includeType = False, includeOptional = True, functionOnly = True, typeFormat = 'pyBind', addUnderScore = True, pyOnly = False))
    else:
        fnArgumentsOptional = fnArguments
        computeArgumentsOptional = computeArguments
        argumentsOptional = arguments


    generatedLines = []
    generatedLines += ['\n', '// DEF PYTHON BINDINGS\n']
    generatedLines += [f'#define {filePrefix}_pyArguments_t {pyArguments}\n']
    generatedLines += ['// DEF FUNCTION ARGUMENTS\n']
    generatedLines += [f'#define {filePrefix}_functionArguments_t {fnArguments}\n']
    if numOptionals > 0:
        generatedLines += [f'#define {filePrefix}_functionArgumentsOptional_t {fnArgumentsOptional}\n']

    generatedLines += ['// DEF COMPUTE ARGUMENTS\n']
    generatedLines += [f'#define {filePrefix}_computeArguments_t {computeArguments}\n']
    if numOptionals > 0:
        generatedLines += [f'#define {filePrefix}_computeArgumentsOptional_t {computeArgumentsOptional}\n']

    generatedLines += ['// DEF ARGUMENTS\n']
    generatedLines += [f'#define {filePrefix}_arguments_t {arguments}\n']
    if numOptionals > 0:
        generatedLines += [f'#define {filePrefix}_argumentsOptional_t {argumentsOptional}\n']
    generatedLines += [f'#define {filePrefix}_arguments_t_ {arguments_}\n']
    if numOptionals > 0:
        generatedLines += [f'#define {filePrefix}_argumentsOptional_t_ {argumentsOptional_}\n']

    generatedLines += ['\n', '// END PYTHON BINDINGS\n']

    with open(fileName, 'w') as f:
        # f.writelines(prefixLines + ['#pragma region autoGenerated\n'] + generatedLines + ['#pragma endregion\n'] + suffixLines)
        f.writelines(prefixLines + generatedLines + suffixLines)


if __name__ == "__main__":
    # only pass parameters to main()
    process(sys.argv[1:][0])
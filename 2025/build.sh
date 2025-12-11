#!/bin/sh

set -eu

ScriptDirectory="$(dirname "$(readlink -f "$0")")"
cd "$ScriptDirectory"

#- Main
DidWork=0
Build="../build"

clang=1
gcc=0
debug=1
release=0

# Targets
day1=0
day2=0
day3=0
day3_cu=0

# Default build
[ "$#" = 0 ] && day3_cu=1

for Arg in "$@"; do eval "$Arg=1"; done
# Exclusive flags
[ "$release" = 1 ] && debug=0

[ "$debug"   = 1 ] && printf '[debug mode]\n'
[ "$release" = 1 ] && printf '[release mode]\n'
mkdir -p "$Build"

CU_Compile()
{
 Source="$1"
 Out="$2"

 Compiler=nvcc
 printf '[%s compile]\n' "$Compiler"

 Flags="
 -g -G -I$ScriptDirectory -DOS_LINUX=1
 -arch sm_50
 "
 WarningFlags="
 -diag-suppress 1143
 -diag-suppress 2464
 -diag-suppress 177
 -diag-suppress 550
 -Wno-deprecated-gpu-targets
 -Xcompiler -Wall
 -Xcompiler -Wextra
 -Xcompiler -Wconversion
 -Xcompiler -Wdouble-promotion

 -Xcompiler -Wno-pointer-arith
 -Xcompiler -Wno-attributes 
 -Xcompiler -Wno-unused-but-set-variable 
 -Xcompiler -Wno-unused-variable 
 -Xcompiler -Wno-write-strings
 -Xcompiler -Wno-pointer-arith
 -Xcompiler -Wno-unused-parameter
 -Xcompiler -Wno-unused-function
 -Xcompiler -Wno-missing-field-initializers
 "
 Flags="$Flags $WarningFlags"

 DebugFlags="-g -G -DAOC_INTERNAL=1"
 ReleaseFlags="-O3"

[ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
[ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"

 printf '%s\n' "$Source"
 Source="$(readlink -f "$Source")"
 $Compiler $Flags "$Source" -o "$Build"/"$Out"

 DidWork=1
}

C_Compile()
{
 Source="$1"
 Out="$2"

 [ "$gcc"   = 1 ] && Compiler="g++"
 [ "$clang" = 1 ] && Compiler="clang"
 printf '[%s compile]\n' "$Compiler"
 
 CommonCompilerFlags="-DOS_LINUX=1 -fsanitize-trap -nostdinc++ -I$ScriptDirectory"
 CommonWarningFlags="-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-sign-conversion -Wno-sign-compare -Wno-double-promotion -Wno-unused-but-set-variable -Wno-unused-variable -Wno-write-strings -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers"
 LinkerFlags=""

 DebugFlags="-g -ggdb -g3 -DAOC_INTERNAL=1"
 ReleaseFlags="-O3"

 ClangFlags="-fdiagnostics-absolute-paths -ftime-trace
-Wno-null-dereference -Wno-missing-braces -Wno-vla-extension -Wno-writable-strings   -Wno-address-of-temporary -Wno-int-to-void-pointer-cast"

 GCCFlags="-Wno-cast-function-type -Wno-missing-field-initializers -Wno-int-to-pointer-cast"

 Flags="$CommonCompilerFlags"
 [ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
 [ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"
 Flags="$Flags $CommonWarningFlags"
 [ "$clang" = 1 ] && Flags="$Flags $ClangFlags"
 [ "$gcc"   = 1 ] && Flags="$Flags $GCCFlags"
 Flags="$Flags $LinkerFlags"

 printf '%s\n' "$Source"
 $Compiler $Flags "$(readlink -f "$Source")" -o "$Build"/"$Out"

 DidWork=1
}

[ "$day1"    = 1 ] && C_Compile  ./day1/day1.c  day1
[ "$day2"    = 1 ] && C_Compile  ./day2/day2.c  day2
[ "$day3"    = 1 ] && C_Compile  ./day3/day3.c  day3
[ "$day3_cu" = 1 ] && CU_Compile ./day3/day3.cu day3_cu

if [ "$DidWork" = 0 ]
then
 printf 'ERROR: No valid build target provided.\n'
 printf 'Usage: %s <day1/day2/day3/day3_cu>\n' "$0"
fi

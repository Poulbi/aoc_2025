#!/bin/sh

set -eu

ScriptDirectory="$(dirname "$(readlink -f "$0")")"
cd "$ScriptDirectory"

#- Globals
CommonCompilerFlags="-DOS_LINUX=1 -fsanitize-trap -nostdinc++ -I./lib"
CommonWarningFlags="-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-sign-conversion -Wno-sign-compare -Wno-double-promotion -Wno-unused-but-set-variable -Wno-unused-variable -Wno-write-strings -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function"
LinkerFlags=""

DebugFlags="-g -ggdb -g3"
ReleaseFlags="-O3"

ClangFlags="-fdiagnostics-absolute-paths -ftime-trace
-Wno-null-dereference -Wno-missing-braces -Wno-vla-extension -Wno-writable-strings -Wno-missing-field-initializers -Wno-address-of-temporary -Wno-int-to-void-pointer-cast"

GCCFlags="-Wno-cast-function-type -Wno-missing-field-initializers -Wno-int-to-pointer-cast"

#- Main
Build="../build"

clang=1
gcc=0
debug=1
release=0

day1=0
day2=0

for Arg in "$@"; do eval "$Arg=1"; done
# Exclusive flags
[ "$release" = 1 ] && debug=0
[ "$gcc"     = 1 ] && clang=0

[ "$gcc"   = 1 ] && Compiler="g++"
[ "$clang" = 1 ] && Compiler="clang"

Flags="$CommonCompilerFlags"
[ "$debug"   = 1 ] && Flags="$Flags $DebugFlags"
[ "$release" = 1 ] && Flags="$Flags $ReleaseFlags"
Flags="$Flags $CommonCompilerFlags"
Flags="$Flags $CommonWarningFlags"
[ "$clang" = 1 ] && Flags="$Flags $ClangFlags"
[ "$gcc"   = 1 ] && Flags="$Flags $GCCFlags"
Flags="$Flags $LinkerFlags"

[ "$debug"   = 1 ] && printf '[debug mode]\n'
[ "$release" = 1 ] && printf '[release mode]\n'
printf '[%s compile]\n' "$Compiler"

mkdir -p "$Build"

DidWork=0
Compile()
{
 Source="$1"
 Out="$2"
 printf '%s\n' "$Source"
 $Compiler $Flags -o "$Build"/"$Out" "$Source"
 DidWork=1
}

[ "$day1"    = 1 ] && Compile ./day1/day1.c day1
[ "$day2"    = 1 ] && Compile ./day2/day2.c day2

if [ "$DidWork" = 0 ]
then
 printf 'ERROR: No build provided.\n'
 printf 'Usage: %s <day1>\n' "$0"
fi

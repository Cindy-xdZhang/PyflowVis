@echo off

if not exist "build" (
    mkdir "build"
)

cd "build"

cmake ..

:: check CMake success or not
if %errorlevel% equ 0 (
    echo CMake build succeeded.
    MSBuild example.vcxproj /p:Configuration=Debug /p:Platform=x64

    :: check Debug exists or not
    if exist "Debug" (
        :: copy build/Debug/ *.pyd to ../Pyds 
        for %%F in ("Debug\*.pyd") do (
            copy "%%F" "..\Pyds\"
            echo Copied "%%F" to "..\Pyds\"
    
        )
    ) else (
        echo Debug directory not found.
    )
) else (
    echo CMake build failed.
)

:: 
cd ..

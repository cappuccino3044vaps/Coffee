@echo off
set source=%1
set dest=%2

for %%f in ("%source%\*.*") do (
    set "base=%%~nf"
    set "ext=%%~xf"
    
    if exist "%dest%\%%~nxf" (
        set /a counter=2
        :loop
        if exist "%dest%\!base!_!counter!!ext!" (
            set /a counter+=1
            goto :loop
        )
        move "%%f" "%dest%\!base!_!counter!!ext!"
        echo Moved %%~nxf to !base!_!counter!!ext!
    ) else (
        move "%%f" "%dest%"
        echo Moved %%~nxf
    )
)
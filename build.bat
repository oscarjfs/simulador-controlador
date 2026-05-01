@echo off
REM Build script for Windows executable
echo Compilando Simulador Controlador con PyInstaller...

REM Install pyinstaller if not present
pip install pyinstaller

REM Build the executable
pyinstaller simulador_controlador.spec --clean

echo.
echo Compilacion completada.
echo El ejecutable se encuentra en la carpeta dist\
pause
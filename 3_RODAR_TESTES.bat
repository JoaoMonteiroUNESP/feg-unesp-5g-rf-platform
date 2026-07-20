@echo off
chcp 65001 > nul
title FEG-UNESP RF Research - Testes

echo ================================================================
echo   FEG-UNESP RF Research Platform
echo   PASSO 3 (opcional) - Suite de testes (pytest)
echo ================================================================
echo.

cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERRO] Ambiente virtual nao encontrado.
    echo Execute primeiro: 1_INSTALAR.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Executando a suite de testes. Os testes de ML podem levar alguns minutos.
echo.
python -X utf8 -m pytest -v

echo.
echo ================================================================
echo   Testes finalizados.
echo   Procure por "passed" (verde) e "failed" (vermelho) acima.
echo ================================================================
echo.
pause

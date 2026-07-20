@echo off
chcp 65001 > nul
title FEG-UNESP RF Research - Instalacao

echo ================================================================
echo   FEG-UNESP RF Research Platform
echo   PASSO 1 de 3 - Instalacao de dependencias
echo ================================================================
echo.

REM Garante que estamos na pasta do script
cd /d "%~dp0"

REM ----- Verifica Python -----
where python > nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    echo.
    echo Instale Python 3.11 ou superior em:
    echo     https://www.python.org/downloads/
    echo.
    echo IMPORTANTE: marque a caixa "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)

echo Python detectado:
python --version
echo.

REM ----- Cria ambiente virtual se nao existir -----
if not exist venv\ (
    echo Criando ambiente virtual em .\venv\ ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual.
        pause
        exit /b 1
    )
) else (
    echo Ambiente virtual ja existe; reaproveitando.
)
echo.

REM ----- Ativa venv e instala -----
call venv\Scripts\activate.bat
echo Atualizando pip...
python -m pip install --quiet --upgrade pip
echo.
echo Instalando dependencias do projeto (pode demorar de 2 a 6 minutos
echo na primeira vez, dependendo da conexao)...
echo.
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao instalar dependencias. Verifique sua conexao
    echo com a internet e tente novamente.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Instalacao concluida com sucesso.
echo.
echo   Proximo passo: clique duas vezes em
echo       2_INICIAR_DASHBOARD.bat
echo ================================================================
echo.
pause

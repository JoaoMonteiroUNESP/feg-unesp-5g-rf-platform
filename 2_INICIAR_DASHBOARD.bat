@echo off
chcp 65001 > nul
title FEG-UNESP RF Research - Dashboard

echo ================================================================
echo   FEG-UNESP RF Research Platform
echo   PASSO 2 de 3 - Iniciando o dashboard
echo ================================================================
echo.

cd /d "%~dp0"

REM ----- Confere venv -----
if not exist venv\Scripts\activate.bat (
    echo [ERRO] Ambiente virtual nao encontrado.
    echo.
    echo Execute primeiro o passo 1: clique duas vezes em
    echo     1_INSTALAR.bat
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo O dashboard sera aberto automaticamente no seu navegador padrao
echo em ~6 segundos.
echo.
echo Endereco: http://127.0.0.1:8000
echo.
echo IMPORTANTE:
echo   - Para PARAR o servidor: pressione Ctrl+C nesta janela.
echo   - Se o navegador nao abrir, copie o endereco acima e cole nele.
echo.
echo ================================================================
echo.

REM Agendamento: abre o browser em background apos 6s
start "" /B cmd /c "timeout /t 6 /nobreak > nul && start http://127.0.0.1:8000"

REM Inicia o servidor (bloqueante; encerra com Ctrl+C)
python -X utf8 -m uvicorn app.main:app --port 8000 --host 127.0.0.1

echo.
echo Servidor encerrado.
pause

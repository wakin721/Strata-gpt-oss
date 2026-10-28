@echo off
setlocal
chcp 65001 >nul
set "VSLANG=1033"
set "GPT_OSS_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%GPT_OSS_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "GPT_OSS_VS=%%i"
if not defined GPT_OSS_VS exit /b 1
call "%GPT_OSS_VS%\VC\Auxiliary\Build\vcvars64.bat"
if not "%errorlevel%"=="0" exit /b 1
set "GPT_OSS_CMAKE=%GPT_OSS_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "GPT_OSS_CTEST=%GPT_OSS_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
pushd "%~dp0..\.."
"%GPT_OSS_CMAKE%" --fresh -S . -B build-gpt-oss-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_DEFAULT_CMP0141=NEW -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded -DSTRATA_GPT_OSS_PROTOTYPE_ONLY=ON -DSTRATA_GPT_OSS_GGML_DIR="%~1" -DSTRATA_GPT_OSS_TEST_PTX="%~2"
if not "%errorlevel%"=="0" goto failed
"%GPT_OSS_CMAKE%" --build build-gpt-oss-ninja --clean-first --parallel 4
if not "%errorlevel%"=="0" goto failed
"%GPT_OSS_CTEST%" --test-dir build-gpt-oss-ninja --output-on-failure
if not "%errorlevel%"=="0" goto failed
popd
exit /b 0
:failed
popd
exit /b 1
